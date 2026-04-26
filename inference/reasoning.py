import numpy as np
from core.bind import bind, unbind, normalize, cosine, DIM
from core.encoder import HolisticEncoder
from core.registry import EntityRegistry

REFINEMENT_STEPS = 8
CONVERGENCE_EPS  = 1e-4
QUERY_DECAY      = 0.70
CHAIN_THRESHOLD  = 0.20
MAX_CHAIN_DEPTH  = 3


class ThoughtStep:
    def __init__(self, step: int, vector: np.ndarray, top_concepts: list[tuple[str, float]]):
        self.step         = step
        self.vector       = vector
        self.top_concepts = top_concepts

    def __repr__(self):
        concepts = ", ".join(f"{c}({s:.2f})" for c, s in self.top_concepts[:3])
        return f"[{self.step}] {concepts}"


class ReasoningEngine:
    def __init__(self, encoder: HolisticEncoder, registry: EntityRegistry):
        self.encoder  = encoder
        self.registry = registry

    def _top_concepts(self, v: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        scores = []
        for concept, vec in self.encoder._concepts.items():
            s = cosine(v, vec)
            scores.append((concept.lower(), s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def compose(self, *surfaces: str) -> np.ndarray:
        vecs = []
        for surface in surfaces:
            entity = self.registry.get(surface)
            if entity is not None:
                vecs.append(entity.v)
            else:
                vecs.append(normalize(self.encoder.encode(surface)))
        if not vecs:
            return np.zeros(DIM)
        if len(vecs) == 1:
            return vecs[0]
        result = vecs[0]
        for v in vecs[1:]:
            result = normalize(bind(result, v))
        return result

    def decompose(self, compound: np.ndarray, *surfaces: str) -> np.ndarray:
        result = compound
        for surface in surfaces:
            entity = self.registry.get(surface)
            v = entity.v if entity is not None else normalize(self.encoder.encode(surface))
            result = normalize(unbind(result, v))
        return result

    def refine(self, v: np.ndarray, entity_surface: str,
               query_vec: np.ndarray = None) -> tuple[np.ndarray, list[ThoughtStep]]:
        entity = self.registry.get(entity_surface)
        if entity is None:
            return v, []

        trace   = []
        current = v.copy()
        decay   = 1.0

        for step in range(REFINEMENT_STEPS):
            prev = current.copy()

            if query_vec is not None:
                blended = normalize(bind(current, query_vec * decay))
                decay  *= QUERY_DECAY
            else:
                blended = current

            entity._ensure_W()
            raw  = np.tanh(entity.W @ blended)
            norm = np.linalg.norm(raw)
            if norm < 1e-6:
                break
            current = raw / norm

            top = self._top_concepts(current)
            trace.append(ThoughtStep(step, current.copy(), top))

            if float(np.linalg.norm(current - prev)) < CONVERGENCE_EPS:
                break

        return current, trace

    def reason(self, query: str,
               entity_surfaces: list[str]) -> tuple[np.ndarray, list[ThoughtStep]]:
        query_vec = normalize(self.encoder.encode(query))

        if not entity_surfaces:
            best_score, best_surface = -1.0, None
            for e in self.registry.all():
                s = cosine(query_vec, e.v)
                if s > best_score:
                    best_score, best_surface = s, e.surface
            if best_surface is None:
                return query_vec, []
            entity_surfaces = [best_surface]

        if len(entity_surfaces) == 1:
            e = self.registry.get(entity_surfaces[0])
            base = normalize(bind(query_vec, e.v)) if e else query_vec
            return self.refine(base, entity_surfaces[0], query_vec=query_vec)
        else:
            composed = self.compose(*entity_surfaces)
            base     = normalize(bind(query_vec, composed))
            best_vec, best_trace, best_score = None, [], -1.0
            for surface in entity_surfaces:
                vec, trace = self.refine(base, surface, query_vec=query_vec)
                score = float(cosine(vec, query_vec))
                if score > best_score:
                    best_vec, best_trace, best_score = vec, trace, score
            return best_vec, best_trace

    def thought_trace_str(self, trace: list[ThoughtStep]) -> str:
        lines = []
        for step in trace:
            concepts = " -> ".join(c for c, _ in step.top_concepts[:4])
            lines.append(f"  [{step.step}] {concepts}")
        return "\n".join(lines)

    def infer_from_chain(
        self,
        chain_nodes: list,
        query_vec: np.ndarray,
        broca,
        top_k: int = 3,
    ) -> list[str]:
        inferred: list[str] = []
        seen: set[str] = set()

        for max_depth in (1, 2, 3):
            nodes_at_depth = [n for n in chain_nodes if n.depth == max_depth]
            for node in nodes_at_depth[:4]:
                path = node.path
                if not path:
                    continue
                facts_per_hop = []
                valid = True
                for surface in path:
                    entity = self.registry.get(surface)
                    if entity is None:
                        valid = False
                        break
                    scored = broca._score_facts(entity, query_vec, top_k=2)
                    if not scored or scored[0][0] < CHAIN_THRESHOLD:
                        valid = False
                        break
                    facts_per_hop.append(scored[0][1])
                if not valid or not facts_per_hop:
                    continue
                combined = " — ".join(facts_per_hop)
                if combined not in seen:
                    seen.add(combined)
                    inferred.append(combined)
                if len(inferred) >= top_k:
                    return inferred

        return inferred