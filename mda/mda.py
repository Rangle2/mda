"""
MDA — Modular Dynamic Architecture
Token-free, entity-centric, online learning system.
"""

import numpy as np
from pathlib import Path
from mda.core.encoder import HolisticEncoder
from mda.core.registry import EntityRegistry
from mda.training.checkpoint import save as _save, load as _load
from mda.inference.broca import BrocaModule
from mda.inference.memory import ConversationMemory
from mda.inference.translator import MDATranslator
from mda.inference.associative import AssociativeChain
from mda.core.bind import normalize
from mda.core.event import EventFrame, time_encode, event_cosine


def _load_stopwords(filename: str) -> frozenset:
    path = Path(__file__).parent / "data" / filename
    if not path.exists():
        return frozenset()
    words: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return frozenset(words)


class MDA:
    def __init__(self, dim: int = 512):
        self.dim      = dim
        self.encoder  = HolisticEncoder(dim)
        self.registry = EntityRegistry(encoder=self.encoder, dim=dim)
        self.broca    = BrocaModule(self.encoder, self.registry)
        self._memory     = ConversationMemory()
        self._translator = MDATranslator()
        self._chain      = AssociativeChain(self.registry, self.encoder)
        self._turn_count:   int            = 0
        self._session_meta: dict           = {}
        self._history:      list           = []
        # EventFrame store: list of (timestamp, frame, description_text)
        self._event_store: list[tuple[float, EventFrame, str]] = []

    def load(self, path: str, streaming: bool = False,
             max_entities: int | None = None) -> "MDA":
        self._translator.load_cache("data/tr_cache.json")
        self._load_streaming(path, max_entities)
        n  = self.registry.count()
        vs = self.broca.vocab_size()
        print(f"Dataset loaded: {n:,} entities")
        print(f"Broca vocab: {vs['fact_store']} facts")
        return self

    def _load_streaming(self, path: str,
                        max_entities: int | None = None) -> None:
        total = 0

        try:
            import ijson
            def _iter():
                with open(path, "rb") as f:
                    try:
                        yield from ijson.items(f, "entities.item")
                        return
                    except Exception:
                        pass
                with open(path, "rb") as f:
                    yield from ijson.items(f, "item")
            source = _iter()
        except ImportError:
            import json
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            records = raw if isinstance(raw, list) else raw.get("entities", [])
            source  = iter(records)

        for record in source:
            if max_entities and total >= max_entities:
                break
            surface  = record.get("surface", "")
            category = record.get("category", "unknown")
            if not surface or len(surface.strip()) < 2:
                continue
            if surface.lower().strip() in self._LOOKUP_STOPWORDS:
                continue
            entity   = self.registry.get_or_create(surface, category)
            self.encoder.register_concept(surface, category)
            facts    = record.get("facts",    [])
            facts_en = record.get("facts_en", []) or facts
            self.broca.learn_from_facts(entity.id, facts_en[:10])
            self.broca.store_facts(entity.id, facts[:10])
            total += 1
            if total % 10_000 == 0:
                print(f"  Loaded {total:,} entities...", end="\r")

        print(f"  Loaded {total:,} entities total        ")

    def teach(self, surface: str, facts: list[str], category: str = "unknown") -> "MDA":
        from mda.core.bind import bind_many
        entity = self.registry.get_or_create(surface, category)
        for fact in facts:
            av = self.encoder.encode(fact)
            entity.a = av
            S = bind_many(entity.v, entity.r, av)
            entity.update_memory(S * av)
            entity.update_W(av, role="agent")
            entity.use_count += 1
            entity.update_beta(float(np.mean(np.abs(av))))
            entity.decay()
            entity.grow(self.encoder)
        self.broca.learn_from_facts(entity.id, facts)
        self.broca.store_facts(entity.id, facts)
        return self

    def relate(self, surface1: str, surface2: str) -> "MDA":
        e1 = self.registry.get(surface1)
        e2 = self.registry.get(surface2)
        if e1 and e2:
            e1.update_relation(e2.id, e2.v, 0.5)
            e2.update_relation(e1.id, e1.v, 0.5)
            from mda.core.bind import bind
            e1.add_synapse(e2, bind)
            e2.add_synapse(e1, bind)
        return self

    def record_event(
        self,
        agent:      str | None = None,
        verb:       str | None = None,
        patient:    str | None = None,
        location:   str | None = None,
        instrument: str | None = None,
        cause:      str | None = None,
        result:     str | None = None,
        t:          float | None = None,
        step:       int | None = None,
    ) -> "MDA":
        """Record a structured event into the event store.

        Each named argument is a surface string or None.  The corresponding
        entity vector is looked up (or encoded on the fly); missing slots are
        left as None in the EventFrame.  Pass *t* (Unix timestamp) and/or
        *step* (turn counter) to attach temporal encoding to the time slot.

        Example::

            mda.record_event(agent="Alice", verb="builds", patient="MDA")
        """
        import time as _time

        def _slot_vec(surface: str | None) -> np.ndarray | None:
            if surface is None:
                return None
            ent = self.registry.get(surface)
            if ent is not None:
                return ent.v.copy()
            return normalize(self.encoder.encode(surface))

        t_vec: np.ndarray | None = None
        if t is not None or step is not None:
            t_val = t if t is not None else _time.time()
            t_vec = time_encode(t_val, step=step)

        frame = EventFrame(
            agent      = _slot_vec(agent),
            verb       = _slot_vec(verb),
            patient    = _slot_vec(patient),
            location   = _slot_vec(location),
            instrument = _slot_vec(instrument),
            cause      = _slot_vec(cause),
            result     = _slot_vec(result),
            time       = t_vec,
        )

        parts = [s for s in [agent, verb, patient, location] if s]
        desc  = " | ".join(parts) if parts else "(unnamed event)"
        self._event_store.append((_time.time(), frame, desc))

        # Rolling window: keep at most 500 events
        if len(self._event_store) > 500:
            self._event_store = self._event_store[-500:]

        return self

    def encode(self, text: str) -> np.ndarray:
        return normalize(self.encoder.encode(text))

    def find_similar(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        from mda.core.bind import cosine
        vec    = normalize(self.encoder.encode(text))
        scores = [(e.surface, cosine(vec, e.v)) for e in self.registry.all()]
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def confidence(self, surface: str, query: str) -> float:
        """Return how confident MDA is about an entity given a query. 0.0–1.0."""
        entity = self.registry.get(surface)
        if entity is None:
            return 0.0
        return self.broca.confidence(entity.id, query)

    def save(self, path: str, user_id: str = "default",
             model_name: str = "unknown") -> dict:
        meta = _save(self.registry, self.broca, path,
                     user_id=user_id, model_name=model_name,
                     turn_count=self._turn_count)
        self._translator.save_cache("data/tr_cache.json")
        self._session_meta = meta
        return meta

    @classmethod
    def from_checkpoint(cls, path: str) -> "MDA":
        model = cls()
        meta = _load(model.registry, model.broca, path)
        model._session_meta = meta
        model._turn_count   = meta.get("turn_count", 0)
        return model

    def process(self, text: str) -> None:
        from mda.core.bind import bind_many
        words = text.strip().split()
        if not words:
            return
        entities_found = []
        for w in words:
            e = self.registry.get(w)
            if e:
                entities_found.append(e)
            elif w[0].isupper() if w else False:
                cat = self.registry.infer_category(self.encoder.encode(text))
                entities_found.append(self.registry.get_or_create(w, cat))
        av = self.encoder.encode(text)
        for entity in entities_found:
            entity.a = av.copy()
            S = bind_many(entity.v, entity.r, av)
            entity.update_memory(S * av)
            entity.update_W(av)
            entity.use_count += 1
            entity.update_beta(float(np.mean(np.abs(av))))
            entity.decay()
            entity.grow(self.encoder)
        if len(entities_found) > 1:
            self.registry.update_synapses_all(entities_found, av)

    def experience(self, text: str) -> None:
        from mda.core.bind import normalize, bind

        en_text = self._translator.to_english(text)
        words   = en_text.strip().split()
        active_entities = []
        for word in words:
            e = self.registry.get(word)
            if e:
                active_entities.append(e)

        if not active_entities:
            return

        input_vec = normalize(self.encoder.encode(en_text))

        for entity in active_entities:
            activation = entity.ensemble_activation(input_vec)
            entity.a = activation
            entity.use_count += 1
            entity.update_beta(float(np.mean(np.abs(activation))))
            entity.decay()
            entity.grow(self.encoder)

        if len(active_entities) > 1:
            for entity in active_entities:
                others = [e for e in active_entities if e is not entity]
                entity.update_synapses(others, input_vec)

        for entity in active_entities:
            for neuron in entity.neurons:
                neuron.hebbian_update(input_vec)

        self._memory.add(
            role="experience",
            text=en_text,
            vector=input_vec,
            entities=[e.surface for e in active_entities],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Stopwords — loaded from txt files under data/. Edit the files to add/remove words, no code change needed.
    _LOOKUP_STOPWORDS  = _load_stopwords("stopwords.txt")
    _CONTENT_STOPWORDS = _load_stopwords("stopwords_content.txt")
    _VERB_STOPWORDS    = frozenset({
        # build family
        "built", "build", "builds", "building",
        # create family
        "created", "create", "creates", "creating",
        # develop family
        "developed", "develop", "develops", "developing",
        # found / launch / start
        "founded", "found", "founds", "founding",
        "started", "start", "starts", "starting",
        "launched", "launch", "launches", "launching",
        # design / write / make
        "designed", "design", "designs", "designing",
        "wrote", "write", "writes", "written", "writing",
        "made", "make", "makes", "making",
        # say / name / show
        "called", "named", "shown", "given", "told", "said",
        "calls", "names", "shows", "gives", "tells", "says",
        # work / run / use
        "works", "worked", "work", "working",
        "runs", "running", "run", "ran",
        "used", "uses", "using", "use",
        # implement / deploy / release
        "implemented", "implement", "implements", "implementing",
        "deployed", "deploy", "deploys", "deploying",
        "released", "release", "releases", "releasing",
        # fix / update / extend
        "fixed", "fix", "fixes", "fixing",
        "updated", "update", "updates", "updating",
        "extended", "extend", "extends", "extending",
    })

    # Root verb forms used *only* for event detection in learn().
    # Kept separate from _VERB_STOPWORDS (which is a bootstrap filter)
    # so the two concerns don't couple.  _detect_verb_root() maps any
    # inflected surface form back to one of these roots.
    _VERB_ROOTS: frozenset[str] = frozenset({
        "build", "create", "develop", "found", "start", "launch",
        "design", "write", "make", "call", "name", "show", "give",
        "tell", "say", "work", "run", "use", "implement", "deploy",
        "release", "fix", "update", "extend",
    })

    @staticmethod
    def _detect_verb_root(token: str) -> str | None:
        """Return the canonical root if *token* is a recognised verb form.

        Steps:
          1. Strip all non-alpha characters and lowercase.
          2. Check for exact match in ``_VERB_ROOTS``.
          3. Try common English inflection suffixes in priority order,
             recovering the root with optional silent-e restoration.

        Returns ``None`` when no root is found so the caller can skip
        event recording cleanly.

        Examples::

            _detect_verb_root("building")  -> "build"
            _detect_verb_root("creates")   -> "create"
            _detect_verb_root("designed,") -> "design"
            _detect_verb_root("the")       -> None
        """
        import re as _re
        t = _re.sub(r"[^a-z]", "", token.lower())
        if not t:
            return None
        if t in MDA._VERB_ROOTS:
            return t
        # Suffix table: (suffix_to_strip, optional_char_to_restore)
        # Ordered from longest to shortest to avoid premature short matches.
        _SUFFIXES = [
            ("ying",  "y"),
            ("ied",   "y"),
            ("ies",   "y"),
            ("ying",  ""),
            ("ing",   ""),
            ("tion",  ""),
            ("ed",    ""),
            ("ers",   ""),
            ("er",    ""),
            ("es",    ""),
            ("s",     ""),
        ]
        for suffix, restore in _SUFFIXES:
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                stem = t[: -len(suffix)]
                if stem in MDA._VERB_ROOTS:
                    return stem
                # Silent-e restoration: "creat" -> "create"
                if stem + "e" in MDA._VERB_ROOTS:
                    return stem + "e"
        return None

    def _find_entities_from_text(self, text: str) -> list:
        """Return entity list from text — handles punctuation, case, plural, bigram."""
        import re
        from mda.core.bind import cosine
        found    = []
        seen_ids = set()
        tokens   = text.split()

        def _lookup(s: str):
            clean = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F]", "", s).strip()
            if not clean or len(clean) < 2:
                return None
            if clean.lower() in self._LOOKUP_STOPWORDS:
                return None
            variants = [clean, clean.lower(), clean.title(), clean.upper()]
            cl = clean.lower()
            if cl.endswith("works"):
                variants += [(clean[:-1]).title(), clean[:-1].lower()]
            elif cl.endswith("ies"):
                variants += [(clean[:-3] + "y").title(), (clean[:-3] + "y").lower()]
            elif cl.endswith("s") and len(clean) > 4:
                variants += [(clean[:-1]).title(), clean[:-1].lower()]
            for v in variants:
                e = self.registry.get(v)
                if e:
                    return e
            return None

        for token in tokens:
            e = _lookup(token)
            if e and e.id not in seen_ids:
                found.append(e)
                seen_ids.add(e.id)

        for i in range(len(tokens) - 1):
            bigram = re.sub(
                r"[^a-zA-Z0-9\u00C0-\u024F ]", "",
                tokens[i] + " " + tokens[i + 1]
            ).strip()
            for variant in [bigram, bigram.lower(), bigram.title()]:
                e = self.registry.get(variant)
                if e and e.id not in seen_ids:
                    found.append(e)
                    seen_ids.add(e.id)
                    break

        if not found:
            qv = normalize(self.encoder.encode(text))
            scored = []
            for e in self.registry.all():
                surface_vec = normalize(self.encoder.encode(e.surface))
                scored.append((float(cosine(qv, surface_vec)), e))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][0] > 0.50:
                found.append(scored[0][1])

        return found

    def _background_learn(self, text: str, weight: float = 0.3,
                          store_fact: bool = False) -> None:
        import re
        text     = self._translator.to_english(text)
        entities = self._find_entities_from_text(text)
        seen_ids = {e.id for e in entities}
        for tok in text.split():
            clean = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F]", "", tok).strip()
            cl    = clean.lower()
            if (len(clean) < 2
                    or cl in self._LOOKUP_STOPWORDS
                    or cl in self._CONTENT_STOPWORDS
                    or cl in self._VERB_STOPWORDS):
                continue
            if clean[0].isupper():
                ent = self.registry.get_or_create(clean, "concept")
                if ent.id not in seen_ids:
                    ent.use_count = max(ent.use_count, 3)
                    entities.append(ent)
                    seen_ids.add(ent.id)
            elif clean.isupper() and len(clean) >= 2:
                ent = self.registry.get_or_create(clean, "concept")
                if ent.id not in seen_ids:
                    ent.use_count = max(ent.use_count, 2)
                    entities.append(ent)
                    seen_ids.add(ent.id)
            else:
                ent = self.registry.get(clean) or self.registry.get(clean.title())
                if ent and ent.id not in seen_ids:
                    entities.append(ent)
                    seen_ids.add(ent.id)
        if not entities:
            return
        input_vec = normalize(self.encoder.encode(text))

        for entity in entities:
            entity.add_sense(
                label="prompt",
                context_vec=input_vec,
                context_hints=text.lower().split()[:5],
            )
            entity.update_memory(input_vec)

            entity._ensure_W()
            pred    = np.tanh(entity.W @ entity.v)
            error   = pred - input_vec
            dtanh   = 1.0 - pred ** 2
            grad    = np.outer(error * dtanh, entity.v) + 0.01 * entity.W
            entity.W -= weight * 0.05 * grad
            mx = np.max(np.abs(entity.W))
            if mx > 10.0:
                entity.W /= mx

            for neuron in entity.neurons:
                neuron.hebbian_update(input_vec)

            entity.use_count += 1
            entity.update_beta(float(np.mean(np.abs(input_vec))))
            entity.decay()
            entity.grow(self.encoder)

            if store_fact:
                existing = self.broca._entity_facts.get(entity.id, [])
                if text not in existing:
                    self.broca.learn_from_facts(entity.id, [text])
                    self.broca.store_facts(entity.id, existing + [text])

        if len(entities) > 1:
            self.registry.update_synapses_all(entities, input_vec)

        self._turn_count += 1
        if self._turn_count % 20 == 0:
            self.registry.prune(min_use_count=2, min_synapse_strength=0.05)

    # ------------------------------------------------------------------
    # Public API: learn / prompt
    # ------------------------------------------------------------------

    def learn(self, text: str, source: str = "user",
              _record_event: bool = True) -> "MDA":
        """Process text as explicit learning — updates W, adds senses.

        Pass ``source="load"`` (or ``_record_event=False``) when ingesting
        files or documentation so that conversation memory and the event
        store are not polluted with bulk-loading artefacts.
        """
        if source == "load":
            _record_event = False
        import re
        from mda.core.bind import bind_many
        en_text   = self._translator.to_english(text)
        input_vec = normalize(self.encoder.encode(en_text))
        # recognized_entities: only surfaces that exist in the registry
        # before bootstrapping.  Used for event auto-recording so that
        # bootstrap-derived noun-entities (e.g. "Builds" from "builds")
        # never become event agents or patients.
        recognized_entities = self._find_entities_from_text(en_text)
        entities  = list(recognized_entities)
        seen_ids  = {e.id for e in entities}

        # Bootstrap entities from all content words (new and existing)
        for tok in en_text.split():
            clean = re.sub(r"[^a-zA-Z0-9\u00C0-\u024F]", "", tok).strip()
            cl    = clean.lower()
            if (len(clean) < 3
                    or cl in self._LOOKUP_STOPWORDS
                    or cl in self._CONTENT_STOPWORDS
                    or cl in self._VERB_STOPWORDS):
                continue
            ent = self.registry.get_or_create(clean.title(), "concept")
            if ent.id not in seen_ids:
                entities.append(ent)
                seen_ids.add(ent.id)

        for entity in entities:
            # Build context from neighboring entity vectors
            neighbor_vecs = [
                self.registry.get_by_id(eid).v
                for eid in entity.synapses
                if self.registry.get_by_id(eid) is not None
            ]
            if neighbor_vecs:
                context_v = normalize(
                    np.mean(neighbor_vecs, axis=0) + input_vec
                )
            else:
                context_v = input_vec

            # Update multi-sense or add new sense
            entity.add_sense(
                label=source,
                context_vec=context_v,
                context_hints=en_text.lower().split()[:5],
            )

            # Update W matrix and memory
            entity.update_memory(input_vec)
            entity.update_W(input_vec)
            entity.use_count += 1
            entity.update_beta(float(np.mean(np.abs(input_vec))))
            entity.decay()

            for neuron in entity.neurons:
                neuron.hebbian_update(input_vec)

            entity.grow(self.encoder)

        if len(entities) > 1:
            self.registry.update_synapses_all(entities, input_vec)

        if entities:
            self.broca.learn_from_facts(entities[0].id, [en_text])
            for _e in entities:
                if _e.surface[0].isupper():
                    self.broca.store_facts(_e.id, [en_text])

        # Auto-record a structural event — only for interactive learning,
        # not bulk file ingestion.
        # Quality gates:
        #   1. _record_event must be True (False for source="load")
        #   2. At least 2 registry-recognized entities (not bootstrap nouns)
        #   3. A lemma-matched verb root must be present in the text
        #   4. Agent and patient must each have use_count >= 2 AND
        #      cosine(entity.v, input_vec) >= 0.25 AND not be a bootstrap
        #      noun (use_count == 1 with no synapses)
        if _record_event and len(recognized_entities) >= 2:
            import time as _time
            from mda.core.bind import cosine as _cosine

            # Find the first token whose lemma matches a known verb root.
            verb_root: str | None = None
            for _tok in en_text.split():
                _root = self._detect_verb_root(_tok)
                if _root is not None:
                    verb_root = _root
                    break

            if verb_root is not None:
                _agent   = recognized_entities[0]
                _patient = recognized_entities[1]

                def _is_valid_entity(ent) -> bool:
                    # Reject bootstrap nouns: first-seen, no connections yet.
                    if ent.use_count == 1 and len(ent.synapses) == 0:
                        return False
                    if ent.use_count < 2:
                        return False
                    if float(_cosine(ent.v, input_vec)) < 0.25:
                        return False
                    return True

                if _is_valid_entity(_agent) and _is_valid_entity(_patient):
                    frame = EventFrame(
                        agent   = _agent.v.copy(),
                        verb    = normalize(self.encoder.encode(verb_root)),
                        patient = _patient.v.copy(),
                        time    = time_encode(float(_time.time()),
                                              step=self._turn_count),
                    )
                    parts = [_agent.surface, verb_root, _patient.surface]
                    self._event_store.append(
                        (_time.time(), frame, " | ".join(parts))
                    )
                    if len(self._event_store) > 500:
                        self._event_store = self._event_store[-500:]

        # Only add to conversation memory for interactive learning; bulk
        # loading (source="load") must not pollute the memory summary.
        if source != "load":
            self._memory.add("learn", en_text, input_vec,
                             [e.surface for e in entities])
        return self

    def is_looping(self, threshold: float = 0.90, window: int = 4) -> bool:
        """Return True if the last *window* memory turns are all mutually
        similar above *threshold* — a sign the conversation is stuck in a loop."""
        from mda.core.bind import cosine
        recent = self._memory.recent(window)
        if len(recent) < window:
            return False
        vecs = [m.vector for m in recent]
        scores = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        return all(s >= threshold for s in scores)

    def context_for(self, query: str) -> str:
        """Return memory context string for the given query — no LLM required.

        Activates the entity network related to *query* and returns the
        relevant facts as a multi-line string suitable for injection into
        an LLM prompt.  Returns an empty string when no relevant memory
        is found.
        """
        import time
        from mda.core.bind import cosine

        lines:     list[str] = []
        query_vec  = normalize(self.encoder.encode(query))
        origin     = self._find_entities_from_text(query)

        if not origin:
            # Broad fallback: no entity surface matched the query.
            # Scan the entire fact store with a relaxed threshold so that
            # semantically adjacent facts (different vocabulary, same topic)
            # are still surfaced.
            #
            # Guard: skip facts that contain "MDA" (self-referential system
            # docs) unless the query itself mentions "mda" — prevents system
            # documentation from leaking into unrelated queries.
            _mda_query = "mda" in query.lower()
            scored: list[tuple[float, str]] = []
            seen_f: set[str] = set()
            for entity in self.registry.all():
                for score, fact in self.broca._score_facts(entity, query_vec, top_k=5):
                    if score < 0.10 or fact in seen_f:
                        continue
                    if not _mda_query and "MDA" in fact:
                        continue
                    scored.append((score, fact))
                    seen_f.add(fact)
            scored.sort(key=lambda x: -x[0])
            broad = "\n".join(f"[MEMORY] {f}" for _, f in scored[:5])
            if not broad:
                return ""
            if len(broad) <= 3000:
                return broad
            truncated = broad[:3000]
            last_nl = truncated.rfind("\n")
            return truncated[:last_nl] if last_nl > 0 else truncated

        origin_sim = float(cosine(origin[0].v, query_vec))
        origin_vec = normalize(origin[0].dominant_sense(query_vec))
        origin[0].last_activated = time.time()

        inhibition_threshold = max(0.10, min(0.35, 0.10 + origin_sim * 0.25))

        for score, fact in self.broca._score_facts(origin[0], query_vec, top_k=3):
            if score >= 0.2:
                lines.append(f"[MEMORY] {fact}")

        chain_result = self._chain.expand_from_text(query)

        if chain_result and chain_result.nodes:
            _now = time.time()
            for node in chain_result.nodes[:6]:
                entity = node.entity
                if entity.id == origin[0].id:
                    continue
                focus = float(cosine(entity.v, query_vec))
                topic = float(cosine(entity.v, origin_vec))
                if focus < inhibition_threshold and topic < inhibition_threshold:
                    continue
                entity.last_activated = _now
                top_k = 2 if max(focus, topic) > inhibition_threshold * 1.5 else 1
                for score, fact in self.broca._score_facts(entity, query_vec, top_k=top_k):
                    if score >= 0.2:
                        line = f"[MEMORY] {fact}"
                        if line not in lines:
                            lines.append(line)

            _rel_seen: set[frozenset] = set()
            for node in chain_result.nodes[:6]:
                entity = node.entity
                if float(cosine(entity.v, query_vec)) < inhibition_threshold:
                    continue
                strong = sorted(
                    entity.synapses.items(),
                    key=lambda x: x[1].decayed_strength(_now),
                    reverse=True,
                )[:2]
                for syn_id, syn in strong:
                    if syn.decayed_strength(_now) < 0.3:
                        continue
                    other = self.registry.get_by_id(syn_id)
                    if other:
                        rel_key = frozenset({entity.id, other.id})
                        if rel_key in _rel_seen:
                            continue
                        _rel_seen.add(rel_key)
                        eff = syn.decayed_strength(_now)
                        lines.append(
                            f"[MEMORY] {entity.surface} -> {other.surface} "
                            f"(confidence: {eff:.2f})"
                        )

        # Event memory — gate by cosine relevance to query vector.
        # retrieval_vec() uses un-permuted slot sum so it stays aligned
        # with HolisticEncoder text embeddings (encode() is permuted HDC,
        # which is ~0 cosine with text queries).
        _EVENT_THRESHOLD = 0.20
        from mda.core.bind import cosine as _cosine
        for _ts, frame, desc in reversed(self._event_store[-50:]):
            if _cosine(frame.retrieval_vec(), query_vec) >= _EVENT_THRESHOLD:
                lines.append(f"[EVENT] {desc}")

        seen:  set[str]  = set()
        final: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                final.append(line)
            if len(final) >= 12:
                break

        result = "\n".join(final)
        if len(result) <= 3000:
            return result
        truncated = result[:3000]
        last_nl = truncated.rfind("\n")
        return truncated[:last_nl] if last_nl > 0 else truncated

    def stats(self) -> str:
        lines = [f"Entity count: {self.registry.count()}", self.registry.summary()]
        if self._history:
            last = self._history[-1]
            if isinstance(last, dict):
                lines.append(f"Train: {last['train']:.4f}  Test: {last['test']:.4f}")
            else:
                lines.append(f"Loss: {last:.4f}")
        mem_len = len(self._memory)
        if mem_len:
            lines.append(f"Memory: {mem_len} turns")
        totals = {"total": 0, "alive": 0, "synapses": 0}
        for e in self.registry.all():
            ns = e.neuron_summary()
            totals["total"]    += ns["total"]
            totals["alive"]    += ns["alive"]
            totals["synapses"] += len(e.synapses)
        lines.append(
            f"Neurons: {totals['alive']}/{totals['total']} alive  "
            f"Synapses: {totals['synapses']}"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MDA(dim={self.dim}, entities={self.registry.count()})"
