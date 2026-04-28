from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mda.integrations.loader import Loader
import numpy as np
import requests

from mda import MDA
from mda.inference.reasoning import ReasoningEngine
from mda.core.bind import cosine, normalize
from mda.training.checkpoint import load as _load

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_markdown(text: str) -> str:
    # ## headings → plain text
    text = re.sub(r"#{1,6}\s+", "", text)
    # **bold** and __bold__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    # *italic* and _italic_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # `code`
    text = re.sub(r"`(.+?)`", r"\1", text)
    # collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class MDAEngine:
    def __init__(
        self,
        model: str = "qwen2.5:9b",
        knowledge_path: str = None,
        max_entities: int = None,
        smart_filter: bool = False,
        filter_model: str = "qwen2.5:0.5b",
        user_id: str = "default",
    ) -> None:
        self.mda              = MDA()
        self._reasoning       = ReasoningEngine(self.mda.encoder, self.mda.registry)
        self.model            = model
        self.smart_filter     = smart_filter
        self.filter_model     = filter_model
        self.user_id          = user_id
        self._ollama_url      = "http://localhost:11434/api/chat"
        self._recent_learns:  list[str] = []
        self._last_thinking:  str = ""
        self._last_en_query:  str = ""
        self._last_en_response: str = ""
        self._last_context:   str = ""
        self._loader = None

        if knowledge_path:
            self.mda.load(knowledge_path, streaming=True, max_entities=max_entities)

    @property
    def loader(self) -> "Loader":
        if self._loader is None:
            from mda.integrations.loader import Loader
            self._loader = Loader(self.mda)
        return self._loader

    # ------------------------------------------------------------------
    # Memory paths
    # ------------------------------------------------------------------

    @property
    def _memory_base(self) -> Path:
        base = Path(__file__).parent.parent.parent / ".memory" / self.user_id
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def _session_path(self) -> str:
        slug = self.model.replace(":", "-").replace("/", "-")
        return str(self._memory_base / f"{slug}.mda")

    @property
    def _shared_path(self) -> str:
        return str(self._memory_base / "shared.mda")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self) -> dict:
        # 1. Full session save
        meta = self.mda.save(self._session_path,
                             user_id=self.user_id,
                             model_name=self.model)

        # 2. shared.mda — category="custom" entities only
        custom_ids = {
            eid for eid, e in self.mda.registry._entities.items()
            if e.category == "custom"
        }
        if custom_ids:
            from mda.core.registry import EntityRegistry
            from mda.inference.broca import BrocaModule
            from mda.training.checkpoint import save as _ckpt_save
            tmp_reg   = EntityRegistry()
            tmp_broca = BrocaModule(self.mda.encoder, tmp_reg)
            for eid in custom_ids:
                e  = self.mda.registry._entities[eid]
                ne = tmp_reg.get_or_create(e.surface, e.category)
                ne.v         = e.v.copy()
                ne.h         = e.h.copy()
                ne.use_count = e.use_count
                ne.beta      = e.beta
                if e.W is not None:
                    ne.W = e.W.copy()
                ne.synapses = e.synapses
                if eid in self.mda.broca._entity_facts:
                    tmp_broca._entity_facts[ne.id]     = self.mda.broca._entity_facts[eid]
                    tmp_broca._fact_vecs[ne.id]        = self.mda.broca._fact_vecs.get(eid, [])
                    tmp_broca._entity_positions[ne.id] = self.mda.broca._entity_positions.get(eid, [])
            _ckpt_save(tmp_reg, tmp_broca,
                       self._shared_path.replace(".mda", ""),
                       user_id=self.user_id, model_name="shared",
                       turn_count=self.mda._turn_count)
        return meta

    def _auto_load(self) -> list[str]:
        msgs: list[str] = []
        # 1. Load shared facts (custom entities, all models)
        shared_json = self._shared_path.replace(".mda", ".json")
        if Path(shared_json).exists():
            try:
                _load(self.mda.registry, self.mda.broca,
                      self._shared_path.replace(".mda", ""))
                msgs.append("shared facts loaded")
            except Exception as exc:
                msgs.append(f"shared load failed: {exc}")
        # 2. Load model-specific session
        session_json = self._session_path.replace(".mda", ".json")
        if Path(session_json).exists():
            try:
                meta = _load(self.mda.registry, self.mda.broca,
                             self._session_path.replace(".mda", ""))
                self.mda._session_meta = meta
                self.mda._turn_count   = meta.get("turn_count", 0)
                msgs.append(
                    f"session loaded · turns:{meta.get('turn_count', 0)} "
                    f"· updated:{meta.get('updated_at', '?')[:10]}"
                )
            except Exception as exc:
                msgs.append(f"session load failed: {exc}")
        return msgs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, lang: str = "tr") -> str:
        en_message         = self.mda._translator.to_english(user_message)
        context            = self._build_context(en_message)
        self._last_context = context
        response           = self._call_llm(context, user_message, lang, _en_message=en_message)

        en_response = self._last_en_response
        if self._should_learn(en_response):
            self.mda._background_learn(self._last_en_query,    weight=0.2, store_fact=False)
            self.mda._background_learn(en_response,             weight=0.3, store_fact=True)
            self._recent_learns = (self._recent_learns + [en_response])[-5:]

        if self.mda._turn_count % 10 == 0:
            try:
                self.save()
            except Exception:
                pass

        return response

    def query(self, user_message: str, lang: str = "en") -> str:
        """Read-only query — does not update MDA memory. For evaluation only."""
        en_message         = self.mda._translator.to_english(user_message)
        context            = self._build_context(en_message)
        self._last_context = context
        return self._call_llm(context, user_message, lang, _en_message=en_message)

    def learn(self, text: str) -> None:
        self.mda.learn(text)

    def teach(self, surface: str, facts: list[str],
              category: str = "custom") -> None:
        self.mda.teach(surface, facts, category=category)

    def switch_model(self, new_model: str) -> list[str]:
        """Save current session, reset MDA state, load new model's session."""
        try:
            self.save()
        except Exception:
            pass
        self.model = new_model
        # Reset MDA state so the new model starts clean
        self.mda              = MDA()
        self._reasoning       = ReasoningEngine(self.mda.encoder, self.mda.registry)
        self._recent_learns   = []
        self._last_thinking   = ""
        self._last_en_query   = ""
        self._last_en_response = ""
        self._last_context    = ""
        return self._auto_load()

    def _discover_md_files(self, base: Path) -> list[Path]:
        files: list[Path] = []
        # 1. .memory/ root — global md files
        global_dir = base / ".memory"
        if global_dir.exists():
            files.extend(sorted(global_dir.glob("*.md")))
        # 2. .memory/{user_id}/ — user-specific md files
        user_dir = base / ".memory" / self.user_id
        if user_dir.exists():
            files.extend(sorted(user_dir.glob("*.md")))
        return files

    def _load_md_file(self, path: Path) -> int:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        paragraphs: list[tuple[str, int]] = []
        current_lines: list[str] = []
        current_heading: str = ""
        in_code_block: bool = False
        para_index: int = 0

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if stripped.startswith("---"):
                continue
            if stripped.startswith("|"):
                continue
            if stripped.startswith(">"):
                continue
            if stripped.startswith("#"):
                if current_lines:
                    para_text = " ".join(current_lines)
                    if current_heading:
                        para_text = f"{current_heading}: {para_text}"
                    paragraphs.append((para_text, para_index))
                    para_index += 1
                    current_lines = []
                current_heading = stripped.lstrip("#").strip()
                continue
            if stripped == "":
                if current_lines:
                    para_text = " ".join(current_lines)
                    if current_heading:
                        para_text = f"{current_heading}: {para_text}"
                    paragraphs.append((para_text, para_index))
                    para_index += 1
                    current_lines = []
                continue
            current_lines.append(stripped)

        if current_lines:
            para_text = " ".join(current_lines)
            if current_heading:
                para_text = f"{current_heading}: {para_text}"
            paragraphs.append((para_text, para_index))

        count = 0
        for para_text, idx in paragraphs:
            if len(para_text) < 30:
                continue
            en_para = self.mda._translator.to_english(para_text)
            self.mda.learn(en_para)
            entities = self.mda._find_entities_from_text(en_para)
            for entity in entities:
                if entity.surface[0].isupper():
                    self.mda.broca.store_facts(entity.id, [en_para], positions=[idx])
            count += 1
        return count

    def load_md(self, path: str | None = None) -> int:
        base = Path(__file__).parent.parent.parent
        if path:
            p = Path(path.strip().strip("'\""))
            return self._load_md_file(p)
        files = self._discover_md_files(base)
        count = 0
        for f in files:
            count += self._load_md_file(f)
        return count

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    _SKIP_PHRASES: frozenset[str] = frozenset({
        "No information", "Not enough", "no information",
        "not enough", "unavailable", "unknown",
        "There is not enough", "Not enough information",
    })

    def _is_junk(self, text: str) -> bool:
        tl = text.lower()
        return any(p.lower() in tl for p in self._SKIP_PHRASES)

    def _thought_line(self, entity, query_vec: np.ndarray,
                      threshold: float = 0.30, top_k: int = 5) -> str | None:
        """W matrix → tanh(W @ v) → concept space cosine → '[THOUGHT] surface relates to: ...'
        Returns None if W is absent, zero, or no concepts clear the threshold."""
        if entity.W is None or np.allclose(entity.W, 0):
            return None
        raw  = np.tanh(entity.W @ entity.v)
        norm = float(np.linalg.norm(raw))
        if norm < 1e-6:
            return None
        thought_vec = raw / norm
        scores = [
            (concept, float(cosine(thought_vec, cvec)))
            for concept, cvec in self.mda.encoder._concepts.items()
        ]
        scores.sort(key=lambda x: -x[1])
        concepts = [c for c, s in scores[:top_k] if s >= threshold]
        if not concepts:
            return None
        return f"[THOUGHT] {entity.surface} relates to: {', '.join(c.lower() for c in concepts)}"

    def _build_context(self, user_message: str) -> str:
        lines: list[str] = []

        # Query vector — focal point
        query_vec  = normalize(self.mda.encoder.encode(user_message))
        origin     = self.mda._find_entities_from_text(user_message)
        origin_vec = query_vec  # fallback: use query itself if no origin

        if origin:
            origin_sim = float(cosine(origin[0].v, query_vec))
            origin_vec = normalize(origin[0].dominant_sense(query_vec))
            origin[0].last_activated = time.time()

            # Dynamic inhibition threshold:
            # strong origin (sim→1.0) → threshold→0.35 → narrow/focused context
            # weak origin   (sim→0.0) → threshold→0.10 → broad context
            inhibition_threshold = max(0.10, min(0.35, 0.10 + origin_sim * 0.25))

            for score, fact in self.mda.broca._score_facts(origin[0], query_vec, top_k=3):
                if score >= 0.2 and not self._is_junk(fact):
                    lines.append(f"[MEMORY] {fact}")
        else:
            return ""

        chain_result = self.mda._chain.expand_from_text(user_message)

        if chain_result and chain_result.nodes:
            for node in chain_result.nodes[:6]:
                entity = node.entity

                # Skip if same as origin (already processed)
                if origin and entity.id == origin[0].id:
                    continue

                # Focus score — proximity to query
                focus_score = float(cosine(entity.v, query_vec))
                # Topic score — proximity to origin
                topic_score = float(cosine(entity.v, origin_vec))

                # If both below threshold → inhibit
                if focus_score < inhibition_threshold and \
                   topic_score < inhibition_threshold:
                    continue

                entity.last_activated = time.time()
                best_score = max(focus_score, topic_score)
                # High score → 2 facts, barely above threshold → 1 fact
                top_k = 2 if best_score > inhibition_threshold * 1.5 else 1

                for score, fact in self.mda.broca._score_facts(entity, query_vec, top_k=top_k):
                    if score < 0.2 or self._is_junk(fact):
                        continue
                    line = f"[MEMORY] {fact}"
                    if line not in lines:
                        lines.append(line)

        if chain_result and chain_result.nodes:
            _now = time.time()
            for node in chain_result.nodes[:6]:
                entity = node.entity

                # Focus check for synapse lines too
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
                    other = self.mda.registry.get_by_id(syn_id)
                    if other:
                        eff = syn.decayed_strength(_now)
                        lines.append(
                            f"[MEMORY] {entity.surface} -> {other.surface} "
                            f"(confidence: {eff:.2f})"
                        )

        # [THOUGHT] — W matrix inner representation → concept steering
        # For origin entity + strong chain nodes, max 3 lines
        thought_candidates = [origin[0]] + [
            n.entity for n in (chain_result.nodes if chain_result else [])[:5]
            if n.entity.id != origin[0].id
        ]
        thought_count = 0
        for ent in thought_candidates:
            if thought_count >= 3:
                break
            tline = self._thought_line(ent, query_vec)
            if tline and tline not in lines:
                lines.append(tline)
                thought_count += 1

        if chain_result and chain_result.nodes:
            inferred = self._reasoning.infer_from_chain(
                chain_result.nodes,
                query_vec,
                self.mda.broca,
            )
            for inf in inferred:
                if inf not in lines:
                    lines.append(f"[INFERRED] {inf}")

        mem_summary = self.mda._memory.summary()
        if mem_summary.strip() and not self._is_junk(mem_summary):
            lines.append(f"[MEMORY] {mem_summary}")

        seen:  set[str]  = set()
        final: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                final.append(line)
            if len(final) >= 10:
                break

        result = "\n".join(final)
        return result[:3000] if len(result) > 3000 else result

    # ------------------------------------------------------------------
    # Learn gate — 3-layer filter
    # ------------------------------------------------------------------

    def _should_learn(self, text: str) -> bool:
        if len(text.strip()) < 15:
            return False

        _UNCERTAIN = [
            "i don't know", "not sure", "i'm not sure",
            "unclear", "I'm not sure", "cannot say", "I cannot",
        ]
        text_lower = text.lower()
        if any(sig.lower() in text_lower for sig in _UNCERTAIN):
            return False

        if self._recent_learns:
            text_vec = normalize(self.mda.encoder.encode(text))
            for prev in self._recent_learns:
                prev_vec = normalize(self.mda.encoder.encode(prev))
                if float(cosine(text_vec, prev_vec)) > 0.92:
                    return False

        entities = self.mda._find_entities_from_text(text)
        if not entities:
            return True

        total_strength = 0.0
        synapse_count  = 0
        for entity in entities:
            for syn in entity.synapses.values():
                total_strength += syn.strength
                synapse_count  += 1

        if synapse_count == 0 or (total_strength / synapse_count) < 0.3:
            return True

        if self.smart_filter:
            try:
                payload = {
                    "model": self.filter_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Is this a factual statement worth remembering long-term?"
                                " Answer only YES or NO, nothing else.\n\n" + text
                            ),
                        }
                    ],
                    "stream": False,
                }
                resp = requests.post(self._ollama_url, json=payload, timeout=10)
                resp.raise_for_status()
                answer = resp.json()["message"]["content"].strip().upper()
                return "YES" in answer
            except Exception:
                pass  # fall through to default

        entities = self.mda._find_entities_from_text(text)
        return len(entities) > 0

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, context: str, user_message: str, lang: str,
                  _en_message: str | None = None) -> str:
        self._last_thinking    = ""
        self._last_en_query    = ""
        self._last_en_response = ""

        en_message = _en_message or self.mda._translator.to_english(user_message)
        self._last_en_query = en_message

        if context:
            sys_parts: list[str] = [
                "You are a helpful assistant with access to a Modular Dynamic Memory knowledge base.",
                "Respond in the same language as the user's message.",
                "Do not mix languages.",
                "If you are not certain about something, say you don't know instead of guessing.",
                "\n## Modular Dynamic Memory Knowledge Base\n"
                "The facts below come from a trusted Dynamic knowledge base and are MORE RELIABLE "
                "than your general training data. Treat them as ground truth and prioritise them "
                "over anything you learned during training. Do NOT contradict or doubt them.\n"
                "EXCEPTION: If the user explicitly states a correction (e.g. 'the value is now X, not Y'), "
                "accept the user's version and reason from it.\n"
                "Do NOT repeat or quote the facts verbatim — use them to inform a natural answer.\n\n"
            ]
            sys_parts.append(context)
        else:
            sys_parts = [
                "You are a helpful assistant.",
                "Respond in the same language as the user's message.",
                "Do not mix languages.",
                "If you are not certain about something, say you don't know instead of guessing.",
            ]
        messages: list[dict] = [
            {"role": "system", "content": "\n".join(sys_parts)},
            {"role": "user",   "content": user_message},
        ]

        # Dynamic timeout: >=14B → 300s, >=9B → 180s, others → 120s
        _size_match = re.search(r"(\d+)\s*b", self.model, re.IGNORECASE)
        _param_b    = int(_size_match.group(1)) if _size_match else 0
        _timeout    = 300 if _param_b >= 14 else (180 if _param_b >= 9 else 120)

        payload = {
            "model":    self.model,
            "messages": messages,
            "stream":   False,
        }

        try:
            resp = requests.post(self._ollama_url, json=payload, timeout=_timeout)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]

            think_match = _THINK_RE.search(raw)
            self._last_thinking = think_match.group(1).strip() if think_match else ""
            en_clean = _THINK_RE.sub("", raw).strip()
            en_clean = _strip_markdown(en_clean)
            self._last_en_response = en_clean

            return self.mda._translator.to_user_lang(en_clean, target=lang)
        except requests.exceptions.Timeout:
            msg = f"[Ollama timeout — model {self.model!r} took >{_timeout} s]"
            self._last_en_response = msg
            return msg
        except requests.exceptions.ConnectionError:
            msg = "[Ollama not reachable — is it running?]"
            self._last_en_response = msg
            return msg
        except Exception as exc:
            msg = f"[Ollama error: {exc}]"
            self._last_en_response = msg
            return msg


# ---------------------------------------------------------------------------
# AnthropicEngine — cloud model variant with richer context injection
# ---------------------------------------------------------------------------

class AnthropicEngine(MDAEngine):
    """MDAEngine subclass that routes LLM calls to Anthropic Claude.

    Key differences from MDAEngine:
    - Uses Anthropic API instead of local Ollama
    - Richer context injection (more facts, higher limits) suited to cloud models
      with large context windows
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        knowledge_path: str = None,
        max_entities: int = None,
        smart_filter: bool = False,
        user_id: str = "default",
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            knowledge_path=knowledge_path,
            max_entities=max_entities,
            smart_filter=smart_filter,
            user_id=user_id,
        )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Richer context for cloud models
    # ------------------------------------------------------------------

    def _build_context(self, user_message: str) -> str:  # type: ignore[override]
        """Cloud-optimised context: more facts, weaker inhibition, higher limits."""
        lines: list[str] = []

        query_vec  = normalize(self.mda.encoder.encode(user_message))
        origin     = self.mda._find_entities_from_text(user_message)
        origin_vec = query_vec

        if not origin:
            return ""

        origin_sim = float(cosine(origin[0].v, query_vec))
        origin_vec = normalize(origin[0].dominant_sense(query_vec))

        # Weaker inhibition — let more related facts through
        inhibition_threshold = max(0.05, min(0.20, 0.05 + origin_sim * 0.15))

        # More facts for the origin entity
        for score, fact in self.mda.broca._score_facts(origin[0], query_vec, top_k=8):
            if score >= 0.15 and not self._is_junk(fact):
                lines.append(f"[MEMORY] {fact}")

        chain_result = self.mda._chain.expand_from_text(user_message)

        if chain_result and chain_result.nodes:
            for node in chain_result.nodes[:10]:
                entity = node.entity
                if origin and entity.id == origin[0].id:
                    continue

                focus_score = float(cosine(entity.v, query_vec))
                topic_score = float(cosine(entity.v, origin_vec))

                if focus_score < inhibition_threshold and \
                   topic_score < inhibition_threshold:
                    continue

                best_score = max(focus_score, topic_score)
                top_k = 4 if best_score > inhibition_threshold * 1.5 else 2

                for score, fact in self.mda.broca._score_facts(entity, query_vec, top_k=top_k):
                    if score < 0.15 or self._is_junk(fact):
                        continue
                    line = f"[MEMORY] {fact}"
                    if line not in lines:
                        lines.append(line)

        if chain_result and chain_result.nodes:
            _now = time.time()
            for node in chain_result.nodes[:10]:
                entity = node.entity
                if float(cosine(entity.v, query_vec)) < inhibition_threshold:
                    continue
                strong = sorted(
                    entity.synapses.items(),
                    key=lambda x: x[1].decayed_strength(_now),
                    reverse=True,
                )[:3]
                for syn_id, syn in strong:
                    if syn.decayed_strength(_now) < 0.25:
                        continue
                    other = self.mda.registry.get_by_id(syn_id)
                    if other:
                        eff = syn.decayed_strength(_now)
                        lines.append(
                            f"[MEMORY] {entity.surface} -> {other.surface} "
                            f"(confidence: {eff:.2f})"
                        )

        # [THOUGHT] lines
        thought_candidates = [origin[0]] + [
            n.entity for n in (chain_result.nodes if chain_result else [])[:8]
            if n.entity.id != origin[0].id
        ]
        thought_count = 0
        for ent in thought_candidates:
            if thought_count >= 5:
                break
            tline = self._thought_line(ent, query_vec)
            if tline and tline not in lines:
                lines.append(tline)
                thought_count += 1

        if chain_result and chain_result.nodes:
            inferred = self._reasoning.infer_from_chain(
                chain_result.nodes,
                query_vec,
                self.mda.broca,
            )
            for inf in inferred:
                if inf not in lines:
                    lines.append(f"[INFERRED] {inf}")

        mem_summary = self.mda._memory.summary()
        if mem_summary.strip() and not self._is_junk(mem_summary):
            lines.append(f"[MEMORY] {mem_summary}")

        seen:  set[str]  = set()
        final: list[str] = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                final.append(line)
            if len(final) >= 25:
                break

        result = "\n".join(final)
        return result[:4000] if len(result) > 4000 else result

    def query(self, user_message: str, lang: str = "en") -> str:
        """Read-only query — does not update MDA memory. For evaluation only."""
        en_message         = self.mda._translator.to_english(user_message)
        context            = self._build_context(en_message)
        self._last_context = context
        return self._call_llm(context, user_message, lang, _en_message=en_message)

    # ------------------------------------------------------------------
    # Anthropic API call
    # ------------------------------------------------------------------

    def _call_llm(self, context: str, user_message: str, lang: str,
                     _en_message: str | None = None) -> str:
        self._last_thinking    = ""
        self._last_en_query    = ""
        self._last_en_response = ""

        en_message = _en_message or self.mda._translator.to_english(user_message)
        self._last_en_query = en_message

        if not self._api_key:
            msg = "[ANTHROPIC_API_KEY not set]"
            self._last_en_response = msg
            return msg

        try:
            import anthropic
        except ImportError:
            msg = "[anthropic package not installed — run: pip install anthropic]"
            self._last_en_response = msg
            return msg

        if context:
            sys_parts: list[str] = [
                "You are a helpful assistant with access to a Modular Dynamic Memory knowledge base.",
                "Respond in the same language as the user's message.",
                "Do not mix languages.",
                "If you are not certain about something, say you don't know instead of guessing.",
                "\n## Modular Dynamic Memory Knowledge Base\n"
                "The facts below come from a trusted Dynamic knowledge base and are MORE RELIABLE "
                "than your general training data. Treat them as ground truth and prioritise them "
                "over anything you learned during training. Do NOT contradict or doubt them.\n"
                "EXCEPTION: If the user explicitly states a correction (e.g. 'the value is now X, not Y'), "
                "accept the user's version and reason from it.\n"
                "Do NOT repeat or quote the facts verbatim — use them to inform a natural answer.\n\n"
            ]
            sys_parts.append(context)
        else:
            sys_parts = [
                "You are a helpful assistant.",
                "Respond in the same language as the user's message.",
                "Do not mix languages.",
                "If you are not certain about something, say you don't know instead of guessing.",
            ]

        system_content = "\n".join(sys_parts)

        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            resp   = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_content,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = resp.content[0].text

            think_match = _THINK_RE.search(raw)
            self._last_thinking = think_match.group(1).strip() if think_match else ""
            en_clean = _THINK_RE.sub("", raw).strip()
            en_clean = _strip_markdown(en_clean)
            self._last_en_response = en_clean

            return self.mda._translator.to_user_lang(en_clean, target=lang)
        except Exception as exc:
            msg = f"[Anthropic error: {exc}]"
            self._last_en_response = msg
            return msg


# ---------------------------------------------------------------------------
# MDABatchEngine — multi-agent / large LLM context builder
# ---------------------------------------------------------------------------

class MDABatchEngine:
    """Process N queries in a single GPU pass and return N context strings.

    Designed for multi-agent workloads, large LLM context windows, and
    codebase analysis — any scenario where multiple independent queries
    share the same underlying MDA memory.

    Usage::

        engine = MDABatchEngine()
        contexts = engine.build_context_batch([
            "legal contract risk analysis",
            "MDA memory architecture",
        ])
        # contexts: list[str] — one per query, ready for LLM injection

    Single-query path (MDAEngine) is completely unaffected.
    """

    def __init__(
        self,
        model: str = "default",
        dim: int = 512,
        depth: int = 6,
        top_k_branches: int = 5,
        user_id: str = "default",
        knowledge_path: str | None = None,
    ) -> None:
        from mda.core.accelerator import set_mode, MDAMode
        from mda.inference.associative import AssociativeChain

        set_mode(MDAMode.BATCH)

        self.user_id = user_id
        self.model   = model
        self.depth   = depth
        self.top_k   = top_k_branches
        self.mda     = MDA(dim=dim)

        # Memory path — same pattern as MDAEngine
        memory_base = Path(__file__).parent.parent.parent / ".memory" / user_id
        memory_base.mkdir(parents=True, exist_ok=True)

        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model)

        # Load order: model-specific → shared
        for ckpt_path in [str(memory_base / slug), str(memory_base / "shared")]:
            if os.path.exists(ckpt_path + ".json"):
                try:
                    _load(self.mda.registry, self.mda.broca, ckpt_path)
                    print(f"[MDA] loaded {self.mda.registry.count()} entities")
                    break
                except Exception as e:
                    print(f"[MDA] checkpoint error: {e}")

        # Load md files — same pattern as MDAEngine._discover_md_files
        base = Path(__file__).parent.parent.parent
        md_files: list[Path] = []

        global_dir = base / ".memory"
        if global_dir.exists():
            md_files.extend(sorted(global_dir.glob("*.md")))

        user_dir = base / ".memory" / user_id
        if user_dir.exists():
            md_files.extend(sorted(user_dir.glob("*.md")))

        if md_files:
            from mda.integrations.loader import Loader
            loader = Loader(self.mda)
            for md_path in md_files:
                try:
                    count = loader.load_file(str(md_path))
                    print(f"[MDA] loaded {md_path.name} -> {count} facts")
                except Exception as e:
                    print(f"[MDA] md error {md_path.name}: {e}")

        if knowledge_path:
            self.mda.load(knowledge_path)

        # Dedicated instances so batch engine state never bleeds into MDAEngine.
        self._chain  = AssociativeChain(self.mda.registry, self.mda.encoder)
        self._broca  = self.mda.broca
        self._reason = ReasoningEngine(self.mda.encoder, self.mda.registry)

    # ------------------------------------------------------------------
    # Core batch API
    # ------------------------------------------------------------------

    def build_context_batch(self, queries: list[str]) -> list[str]:
        """N queries → N context strings, entity matrix traversed once per batch.

        Steps:
          1. Encode all queries                   — encoder.encode_batch
          2. Find top-k origins per query         — single (N, M) GPU matmul
          3. BFS expansion per query              — CPU, parallel-safe
          4. Batch fact scoring across all nodes  — score_facts_batch (GPU)
          5. Inferred multi-hop paths             — infer_from_chain_batch
          6. Assemble context strings             — CPU
        """
        if not queries:
            return []

        # Step 1 — encode
        query_vecs = self.mda.encoder.encode_batch(queries)   # (N, 512)
        query_vecs = (query_vecs / (
            np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8
        )).astype(np.float32)

        # Step 2 + 3 — parallel BFS
        chain_results = self._chain.expand_batch(
            query_vecs, top_k=self.top_k, max_depth=self.depth
        )

        # Step 4 + 5 — batch fact scoring and path inference
        all_inferred = self._reason.infer_from_chain_batch(
            chain_results, query_vecs, self._broca,
            top_k=self.top_k, max_depth=self.depth,
        )

        # Step 6 — assemble per-query context strings
        contexts: list[str] = []
        for i, (query, chain, inferred) in enumerate(
            zip(queries, chain_results, all_inferred)
        ):
            lines: list[str] = []
            q_vec = query_vecs[i]

            # Origin entity facts
            if chain and chain.nodes:
                origin_entity = chain.nodes[0].entity
                for score, fact in self._broca._score_facts(
                    origin_entity, q_vec, top_k=3
                ):
                    if score >= 0.15:
                        lines.append(f"[MEMORY] {fact}")

            # Chain node facts (skip origin — already handled above)
            if chain and chain.nodes:
                for node in chain.nodes[1:8]:
                    for score, fact in self._broca._score_facts(
                        node.entity, q_vec, top_k=2
                    ):
                        line = f"[MEMORY] {fact}"
                        if score >= 0.15 and line not in lines:
                            lines.append(line)

            # Multi-hop inferred paths
            for inf in inferred:
                lines.append(f"[INFERRED] {inf}")

            ctx = "\n".join(lines[:15])
            contexts.append(ctx[:5000])

        return contexts

    def learn_batch(self, texts: list[str]) -> None:
        """Learn N facts sequentially, then rebuild the entity matrix once."""
        for text in texts:
            self.mda.learn(text)
        self.mda.registry._em_dirty = True
        self.mda.registry._build_entity_matrix()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self) -> None:
        from mda.training.checkpoint import save as _save
        memory_base = Path(__file__).parent.parent.parent / ".memory" / self.user_id
        memory_base.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model)
        path = str(memory_base / slug)
        _save(self.mda.registry, self.mda.broca, path)
        print(f"[MDA] saved {self.mda.registry.count()} entities")

    # ------------------------------------------------------------------
    # Pass-through helpers
    # ------------------------------------------------------------------

    def learn(self, text: str) -> None:
        self.mda.learn(text)

    def teach(self, surface: str, facts: list[str],
              category: str = "custom") -> None:
        self.mda.teach(surface, facts, category=category)


if __name__ == "__main__":
    bridge = MDAEngine(model="qwen2.5:9b", smart_filter=False)
    bridge.learn("Kairfy bir hukuki belge analiz platformudur")
    bridge.teach(
        "Kairfy",
        ["Kairfy processes legal documents", "Kairfy targets Turkish lawyers"],
        category="custom",
    )
    response = bridge.chat("Kairfy nedir?", lang="tr")
    print(response)

    bridge2 = MDAEngine(
        model="qwen2.5:9b",
        smart_filter=True,
        filter_model="qwen2.5:0.5b",
    )
    response2 = bridge2.chat("Python nedir?", lang="tr")
    print(response2)
