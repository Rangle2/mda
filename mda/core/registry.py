import numpy as np
from mda.core.entity import Entity
from mda.core.bind import DIM, random_vector, cosine

_CATEGORY_SEEDS = {
    "person":   42, "animal":  43, "object":  44,
    "place":    45, "concept": 46, "software": 50,
    "ai":       51, "machine": 49, "data":    52,
}

CATEGORY_VECTORS: dict[str, np.ndarray] = {}


def _init_categories() -> None:
    for cat, seed in _CATEGORY_SEEDS.items():
        CATEGORY_VECTORS[cat] = random_vector(DIM, seed=seed)


_init_categories()


def _det_seed(surface: str) -> int:
    key = surface.lower().encode("utf-8")[:8].ljust(8, b"\0")
    return int.from_bytes(key, "little") % (2 ** 31)


class EntityRegistry:
    def __init__(self):
        self._entities: dict[str, Entity]  = {}
        self._surface:  dict[str, str]     = {}
        self._em_matrix: np.ndarray | None = None   # (N, dim) float32, lazy built
        self._em_index:  list[str]          = []     # entity ids in matrix row order
        self._em_dirty:  bool               = True

    def get_or_create(self, surface: str, category: str = "unknown") -> Entity:
        key = surface.lower().strip()
        if key in self._surface:
            e = self._entities[self._surface[key]]
            e.use_count += 1
            return e
        eid    = f"e_{len(self._entities):06d}"
        seed   = _det_seed(surface)
        entity = Entity(id=eid, surface=surface)
        entity.category = category
        if category in CATEGORY_VECTORS:
            rng = np.random.default_rng(seed)
            # Scale noise std so cosine similarity stays consistent regardless of dim
            noise_std = 0.05 * np.sqrt(256 / DIM)
            noise = rng.normal(0, noise_std, DIM)
            from mda.core.bind import normalize
            entity.v = normalize(CATEGORY_VECTORS[category] + noise)
        self._entities[eid]  = entity
        self._surface[key]   = eid
        self._em_dirty = True
        return entity

    def get(self, surface: str) -> Entity | None:
        key = surface.lower().strip()
        if key in self._surface:
            return self._entities[self._surface[key]]
        return None

    def get_by_id(self, eid: str) -> Entity | None:
        return self._entities.get(eid)

    def infer_category(self, action_vec: np.ndarray) -> str:
        best_cat = "unknown"
        best_sim = -1.0
        for cat, v in CATEGORY_VECTORS.items():
            s = cosine(action_vec, v)
            if s > best_sim:
                best_sim = s
                best_cat = cat
        return best_cat if best_sim > 0.1 else "unknown"

    def all(self) -> list[Entity]:
        return list(self._entities.values())

    def count(self) -> int:
        return len(self._entities)

    def summary(self) -> str:
        cats: dict[str, int] = {}
        for e in self._entities.values():
            cats[e.category] = cats.get(e.category, 0) + 1
        lines = [f"Total entities: {self.count()}"]
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:<15} {n}")
        return "\n".join(lines)

    def save_state(self) -> dict:
        return {
            eid: {
                "surface":   e.surface,
                "category":  e.category,
                "use_count": e.use_count,
                "beta":      e.beta,
                "v":         e.v.tolist(),
                "W":         e.W.tolist() if e.W is not None else None,
                "h":         e.h.tolist(),
            }
            for eid, e in self._entities.items()
        }

    def load_state(self, state: dict) -> None:
        import numpy as np
        for eid, data in state.items():
            surface  = data["surface"]
            category = data["category"]
            entity   = self.get_or_create(surface, category)
            entity.use_count = data["use_count"]
            entity.beta      = data["beta"]
            entity.v         = np.array(data["v"])
            entity.W         = np.array(data["W"]) if data.get("W") else None
            entity.h         = np.array(data["h"])

    def update_synapses_all(self, active_entities: list, input_vec) -> None:
        for entity in active_entities:
            others = [e for e in active_entities if e.id != entity.id]
            if others:
                entity.update_synapses(others, input_vec)

    def prune_all(self) -> None:
        for entity in self._entities.values():
            entity.prune()

    def remove(self, entity: Entity) -> None:
        key = entity.surface.lower()
        self._entities.pop(self._surface.get(key, ""), None)
        self._surface.pop(key, None)
        for other in self._entities.values():
            other.synapses.pop(entity.id, None)
        self._em_dirty = True

    def _build_entity_matrix(self) -> None:
        if not self._entities:
            self._em_matrix = None
            self._em_index  = []
            self._em_dirty  = False
            return
        ids  = list(self._entities.keys())
        vecs = np.stack([self._entities[eid].v.astype(np.float32) for eid in ids])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self._em_matrix = vecs / (norms + 1e-8)
        self._em_index  = ids
        self._em_dirty  = False

    def nearest(self, query_vec: np.ndarray, top_k: int = 1
                ) -> list[tuple[float, "Entity"]]:
        """Return top_k entities by cosine similarity using EntityMatrix + batch_cosine."""
        if self._em_dirty or self._em_matrix is None:
            self._build_entity_matrix()
        if self._em_matrix is None:
            return []
        from mda.core.accelerator import batch_cosine
        q = query_vec.astype(np.float32)
        qn = q / (np.linalg.norm(q) + 1e-8)
        scores = batch_cosine(self._em_matrix, qn)
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(float(scores[i]), self._entities[self._em_index[i]]) for i in idx]

    def build_cross_entity_synapses(self, bind_fn, min_use_count: int = 2) -> int:
        """Build synapses between all entity pairs that share common fact keywords.
        Returns number of synapses created."""
        from itertools import combinations
        entities = [e for e in self._entities.values() if e.use_count >= min_use_count]
        created = 0
        for a, b in combinations(entities, 2):
            if b.id not in a.synapses:
                a.add_synapse(b, bind_fn)
                created += 1
        return created

    def prune(
        self,
        min_use_count: int = 2,
        min_synapse_strength: float = 0.05,
    ) -> int:
        to_remove: list[Entity] = []
        for entity in list(self.all()):
            if entity.use_count >= min_use_count:
                continue
            max_strength = (
                max(s.strength for s in entity.synapses.values())
                if entity.synapses else 0.0
            )
            if max_strength < min_synapse_strength:
                to_remove.append(entity)
        for entity in to_remove:
            self.remove(entity)
        return len(to_remove)
