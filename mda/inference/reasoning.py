import numpy as np
from mda.core.bind import bind, unbind, normalize, cosine, DIM
from mda.core.encoder import HolisticEncoder
from mda.core.registry import EntityRegistry

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
        # Collect all candidate paths across depths
        candidate_paths: list[list[str]] = []
        for max_depth in (1, 2, 3):
            for node in [n for n in chain_nodes if n.depth == max_depth][:4]:
                if node.path:
                    candidate_paths.append(node.path)

        if not candidate_paths:
            return []

        return self._infer_paths_parallel(candidate_paths, query_vec, broca, top_k)

    def infer_from_chain_batch(
        self,
        chain_results: list,
        query_vecs: np.ndarray,
        broca,
        top_k: int = 3,
        max_depth: int = 6,
    ) -> list[list[str]]:
        """
        Batch variant: score unique entities once across all N chains,
        then assemble per-chain inferred paths from cache.
        """
        if not chain_results:
            return []

        # 1. Collect all unique surfaces across every chain
        unique_surfaces: set[str] = set()
        for chain in chain_results:
            if chain and chain.nodes:
                for node in chain.nodes:
                    if node.path:
                        unique_surfaces.update(node.path)

        # 2. Score each unique entity once using mean query vector
        mean_q = query_vecs.mean(axis=0).astype(np.float32)
        entity_scores: dict[str, list[tuple[float, str]]] = {}
        for surface in unique_surfaces:
            entity = self.registry.get(surface)
            if entity is not None:
                entity_scores[surface] = broca._score_facts(entity, mean_q, top_k=2)

        # 3. Assemble per-chain results
        all_inferred: list[list[str]] = []
        for i, chain in enumerate(chain_results):
            if not chain or not chain.nodes:
                all_inferred.append([])
                continue

            candidate_paths: list[list[str]] = []
            for max_d in (1, 2, 3):
                for node in [n for n in chain.nodes if n.depth == max_d][:4]:
                    if node.path:
                        candidate_paths.append(node.path)

            inferred: list[str] = []
            seen: set[str] = set()
            for path in candidate_paths:
                facts_per_hop: list[str] = []
                valid = True
                for surface in path:
                    scored = entity_scores.get(surface, [])
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
                    break

            all_inferred.append(inferred)

        return all_inferred

    def _infer_paths_serial(
        self, paths: list, query_vec: np.ndarray, broca, top_k: int
    ) -> list[str]:
        """Original sequential logic, kept for testing equivalence."""
        inferred: list[str] = []
        seen: set[str] = set()
        for path in paths:
            facts_per_hop: list[str] = []
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
                break
        return inferred

    def _infer_paths_parallel(
        self, paths: list, query_vec: np.ndarray, broca, top_k: int
    ) -> list[str]:
        """
        Deduplicated path scoring: each unique entity is scored exactly once.
        No torch dependency — speedup comes from deduplication;
        GPU speedup comes from broca._score_facts internally.
        """
        # 1. Score each unique entity once
        unique_surfaces: set[str] = set()
        for path in paths:
            unique_surfaces.update(path)

        entity_scores: dict[str, list[tuple[float, str]]] = {}
        for surface in unique_surfaces:
            entity = self.registry.get(surface)
            if entity is not None:
                entity_scores[surface] = broca._score_facts(
                    entity, query_vec, top_k=2
                )

        # 2. Assemble paths from cached scores
        inferred: list[str] = []
        seen: set[str] = set()
        for path in paths:
            facts_per_hop: list[str] = []
            valid = True
            for surface in path:
                scored = entity_scores.get(surface, [])
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
                break
        return inferred
