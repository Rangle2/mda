"""
AssociativeChain: starting from one entity, spreads to connected entities
within a semantic boundary. Token-free, attention-free.

Working principle:
  1. Origin entity is activated
  2. Spreads to neighbours via synapses
  3. At each step the cosine distance from the origin vector is checked
  4. Stop if semantic boundary is exceeded
  5. Do not revisit already-visited entities (cycle prevention)
"""

import re
import numpy as np
from dataclasses import dataclass, field
from mda.core.bind import normalize, cosine, bind
from mda.core.entity import Entity
from mda.core.registry import EntityRegistry


SEMANTIC_BOUNDARY = 0.25   # stop when this far from origin
MAX_DEPTH         = 4      # maximum chain depth
MIN_SYNAPSE_STR   = 0.15   # synapses weaker than this are not followed
TOP_K_BRANCHES    = 3      # follow the strongest K synapses at each step


@dataclass
class ChainNode:
    entity:     Entity
    depth:      int
    activation: float
    path:       list  = field(default_factory=list)
    sense_vec:  np.ndarray = None


@dataclass
class ChainResult:
    nodes:         list
    compound_v:    np.ndarray
    origin_v:      np.ndarray
    depth_reached: int

    def active_entities(self) -> list[Entity]:
        return [n.entity for n in self.nodes]

    def activation_map(self) -> dict[str, float]:
        return {n.entity.surface: n.activation for n in self.nodes}


class AssociativeChain:
    def __init__(self, registry: EntityRegistry, encoder):
        self.registry = registry
        self.encoder  = encoder
        self._thresh_cache: tuple | None = None
        self._thresh_entity_count: int   = 0

    def _get_thresholds(self) -> tuple[float, float, float]:
        """
        (dyn_boundary, dyn_syn_min, dyn_query_threshold).
        Pure numpy. Rebuilt only when entity count changes.
        """
        current_count = self.registry.count()
        if (self._thresh_cache is not None
                and self._thresh_entity_count == current_count):
            return self._thresh_cache

        synapse_sims: list[float] = []
        synapse_strs: list[float] = []
        for e in self.registry.all():
            for eid, syn in e.synapses.items():
                nb = self.registry.get_by_id(eid)
                if nb:
                    synapse_sims.append(float(cosine(e.v, nb.v)))
                    synapse_strs.append(syn.strength)

        if synapse_sims:
            dyn_boundary        = max(min(synapse_sims) * 0.8, -0.9)
            dyn_syn_min         = min(synapse_strs) * 0.9
            dyn_query_threshold = min(dyn_boundary, 0.0)
        else:
            dyn_boundary        = -0.5
            dyn_syn_min         = 0.01
            dyn_query_threshold = -0.5

        self._thresh_cache        = (dyn_boundary, dyn_syn_min, dyn_query_threshold)
        self._thresh_entity_count = current_count
        return self._thresh_cache

    def expand(self, origin_entity: Entity,
               context_vec: np.ndarray = None,
               query_vec:   np.ndarray = None) -> ChainResult:
        """
        Expand the chain starting from origin_entity.

        context_vec: context (for multi-sense disambiguation)
        query_vec:   query (determines which direction to expand)

        Thresholds are computed dynamically from actual synapse stats so
        the chain works for sparse spaces (few entities, low synapse strengths).
        """
        if context_vec is None:
            context_vec = origin_entity.v
        if query_vec is None:
            query_vec = origin_entity.v

        origin_v   = normalize(origin_entity.dominant_sense(context_vec))
        visited    = {origin_entity.id}
        queue      = [(0, origin_entity, 1.0, [origin_entity.surface], origin_v)]
        all_nodes  = []
        compound_v = origin_v.copy()

        # Dynamic thresholds — cached, rebuilt only when entity count changes
        dyn_boundary, dyn_syn_min, dyn_query_threshold = self._get_thresholds()

        while queue:
            depth, entity, activation, path, sense_v = queue.pop(0)
            all_nodes.append(ChainNode(
                entity=entity, depth=depth, activation=activation,
                path=path.copy(), sense_vec=sense_v,
            ))

            if depth >= MAX_DEPTH:
                continue

            compound_v = normalize(bind(compound_v, sense_v))

            synapses = sorted(
                entity.synapses.values(), key=lambda s: -s.strength
            )[:TOP_K_BRANCHES]

            for synapse in synapses:
                if synapse.strength < dyn_syn_min:
                    continue
                neighbor = self.registry.get_by_id(synapse.target_id)
                if neighbor is None or neighbor.id in visited:
                    continue
                neighbor_v = normalize(neighbor.dominant_sense(compound_v))
                # Semantic boundary: use raw entity vectors (sense vecs are
                # directional and can anti-correlate with unrelated entities)
                if cosine(neighbor.v, origin_entity.v) < dyn_boundary:
                    continue
                if cosine(neighbor.v, query_vec) < dyn_query_threshold:
                    continue
                visited.add(neighbor.id)
                queue.append((
                    depth + 1, neighbor,
                    activation * synapse.strength * 0.8,
                    path + [neighbor.surface],
                    neighbor_v,
                ))

        all_nodes.sort(key=lambda node: (
            node.depth,
            -node.activation,
            -node.entity.use_count,
        ))

        return ChainResult(
            nodes=all_nodes,
            compound_v=compound_v,
            origin_v=origin_v,
            depth_reached=max((n.depth for n in all_nodes), default=0),
        )

    def expand_from_text(self, text: str,
                         context_vec: np.ndarray = None) -> ChainResult | None:
        """Find entity from text and expand the chain."""
        query_vec = normalize(self.encoder.encode(text))

        origin = None
        for token in text.split():
            clean = re.sub(r"[^\w]", "", token).strip()
            if not clean:
                continue
            entity = (
                self.registry.get(clean) or
                self.registry.get(clean.lower()) or
                self.registry.get(clean.title())
            )
            if entity:
                origin = entity
                break

        if origin is None:
            hits = self.registry.nearest(query_vec, top_k=1)
            if hits and hits[0][0] > 0.25:
                origin = hits[0][1]
            else:
                return None

        return self.expand(
            origin_entity=origin,
            context_vec=context_vec or query_vec,
            query_vec=query_vec,
        )

    def chain_summary(self, result: ChainResult) -> str:
        """Chain summary — context to be passed to BrocaModule."""
        if not result.nodes:
            return ""

        return " -> ".join(
            n.entity.surface for n in
            sorted(result.nodes, key=lambda n: (n.depth, -n.activation))[:5]
        )
