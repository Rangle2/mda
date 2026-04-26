import numpy as np
from mda.core.bind import normalize, cosine
from mda.core.entity import Entity


class BrocaModule:
    def __init__(self, encoder, registry=None):
        self.encoder  = encoder
        self.registry = registry

        self._entity_facts:     dict[str, list[str]]        = {}
        self._fact_vecs:        dict[str, list[np.ndarray]] = {}
        self._entity_positions: dict[str, list[int]]        = {}
        self._categories:       dict[str, str]              = {}
        self._encode_cache:     dict[str, np.ndarray]       = {}

    def _encode(self, text: str) -> np.ndarray:
        if len(self._encode_cache) > 2000:
            self._encode_cache.clear()
        if text not in self._encode_cache:
            self._encode_cache[text] = normalize(self.encoder.encode(text))
        return self._encode_cache[text]

    # ------------------------------------------------------------------
    # Registration (backwards-compat no-ops)
    # ------------------------------------------------------------------

    def register_verb(self, word: str, entity_id: str = None) -> None:
        pass

    def register_property(self, word: str, entity_id: str = None) -> None:
        pass

    def register_domain(self, phrase: str, entity_id: str = None) -> None:
        pass

    def register_category(self, category: str, template: str) -> None:
        self._categories[category] = template

    # ------------------------------------------------------------------
    # Fact storage — pre-encodes vectors at store time
    # ------------------------------------------------------------------

    def store_facts(self, entity_id: str, tr_facts: list[str],
                    positions: list[int] | None = None) -> None:
        _JUNK = {
            "not enough", "no information", "no data", "no details",
            "i don't have", "i don't know", "don't know",
            "cannot find", "can't find", "insufficient",
            "yet.", "about x yet",
        }
        cleaned = [
            f.strip() for f in tr_facts
            if f.strip()
            and not any(j in f.strip().lower() for j in _JUNK)
        ]
        if not cleaned:
            return

        existing_facts = self._entity_facts.get(entity_id, [])
        existing_vecs  = self._fact_vecs.get(entity_id, [])
        existing_pos   = self._entity_positions.get(entity_id, [])

        new_facts = [f for f in cleaned if f not in existing_facts]
        if not new_facts:
            return

        new_vecs = [self._encode(f) for f in new_facts]
        new_pos  = positions if positions is not None else [99] * len(new_facts)

        self._entity_facts[entity_id]     = existing_facts + new_facts
        self._fact_vecs[entity_id]        = existing_vecs  + new_vecs
        self._entity_positions[entity_id] = existing_pos   + new_pos

    def learn_from_facts(self, entity_id: str, facts: list[str],
                         positions: list[int] | None = None) -> None:
        self.store_facts(entity_id, facts, positions=positions)

    # ------------------------------------------------------------------
    # Vocab size
    # ------------------------------------------------------------------

    def vocab_size(self) -> dict:
        return {"fact_store": len(self._entity_facts)}

    # ------------------------------------------------------------------
    # W+fact hybrid scoring
    # ------------------------------------------------------------------

    def _w_concept_score(self, entity: Entity, fact_vec: np.ndarray) -> float:
        """Score a fact vector against the entity's W-activated concept."""
        if entity.W is None or np.allclose(entity.W, 0):
            return 0.0
        w_activated = normalize(np.tanh(entity.W @ entity.v))
        return float(np.dot(w_activated, fact_vec))

    def _score_facts(self, entity: Entity, query_vec: np.ndarray,
                     top_k: int = 3,
                     context_vec: np.ndarray = None) -> list[tuple[float, str]]:
        """Return the top_k facts ranked by a hybrid of query cosine, W score,
        and (when available) sense-aware score."""
        facts = self._entity_facts.get(entity.id, [])
        if not facts:
            return []
        fact_vecs = self._fact_vecs.get(entity.id, [])
        if not fact_vecs:
            return facts[:top_k]
        positions = self._entity_positions.get(entity.id, [])
        scored = []
        for i, (fact, fvec) in enumerate(zip(facts, fact_vecs)):
            query_score = float(np.dot(query_vec, fvec))
            w_score     = self._w_concept_score(entity, fvec)
            if entity.senses and context_vec is not None:
                sense_v     = normalize(entity.dominant_sense(context_vec))
                sense_score = float(np.dot(sense_v, fvec))
                combined    = query_score * 0.35 + w_score * 0.45 + sense_score * 0.20
            elif entity.W is None or np.allclose(entity.W, 0):
                combined = query_score
            else:
                combined = query_score * 0.4 + w_score * 0.6
            scored.append((combined, fact))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def confidence(self, entity_id: str, query: str) -> float:
        entity = self.registry.get_by_id(entity_id)
        if entity is None:
            return 0.0
        facts = self._entity_facts.get(entity_id, [])
        if not facts:
            return 0.0
        q_vec = self._encode(query)
        scored = self._score_facts(entity, q_vec, top_k=1, context_vec=q_vec)
        if not scored:
            return 0.0
        return float(np.clip(scored[0][0], 0.0, 1.0))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, entity: Entity, query_type: str = "nedir",
                 query_text: str = None) -> str:
        name  = entity.surface
        facts = self._entity_facts.get(entity.id, [])

        if not facts:
            return f"Not enough information about {name} yet."

        q_text = query_text if query_text else query_type
        q_vec  = self._encode(q_text)

        # list — bullet format, top-5 by hybrid score
        if query_type == "list":
            items = self._score_facts(entity, q_vec, top_k=5, context_vec=q_vec)
            return f"Known facts about {name}:\n" + "\n".join(f"• {f}" for f in items)

        # keyword sets for specialised filtering
        _VERB_KW  = frozenset(["is ", "provides", "supports", "runs", "enables",
                               "allows", "executes", "implements", "offers",
                               "kullanılır", "çalışır", "sağlar", "sunar",
                               "destekler", "üretir", "işler", "yönetir"])
        _HOWTO_KW = frozenset(["how", " by ", "using", "through", "via"])
        _PAST_KW  = frozenset(["was ", "were ", " had ", "created", "founded",
                               "released", "developed", "designed", "introduced",
                               "launched", "published", "invented", "built"])

        def _matches(fact: str, keywords) -> bool:
            fl = fact.lower()
            return any(kw in fl for kw in keywords)

        # Ranked pool — all facts scored by hybrid metric
        all_facts = self._entity_facts.get(entity.id, [])
        fact_vecs = self._fact_vecs.get(entity.id, [])
        positions = self._entity_positions.get(entity.id, [])
        scored = []
        for i, (fact, fvec) in enumerate(zip(all_facts, fact_vecs)):
            query_score = float(np.dot(q_vec, fvec))
            w_score     = self._w_concept_score(entity, fvec)
            combined    = query_score * (0.4 + w_score * 0.6)
            scored.append((combined, fact))
        scored.sort(key=lambda x: -x[0])

        if query_type in ("ne yapar", "ne yapıyor"):
            preferred = [(s, f) for s, f in scored if _matches(f, _VERB_KW)]
            pool = preferred if preferred else scored

        elif query_type in ("nasıl", "howto"):
            preferred = [(s, f) for s, f in scored if _matches(f, _HOWTO_KW)]
            pool = preferred if preferred else scored

        elif query_type in ("ne yaptı",):
            preferred = [(s, f) for s, f in scored if _matches(f, _PAST_KW)]
            pool = preferred if preferred else scored

        else:  # nedir, kimdir, explain, analyze — and default
            pool = scored

        selected = [f for _, f in pool[:3]]
        return "\n".join(selected)
