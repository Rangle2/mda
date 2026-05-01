"""
Long-Context Accuracy Benchmark v2: MDA vs Incremental ChromaDB RAG
====================================================================
Key difference from v1: RAG now uses ChromaDB with incremental indexing.
Every fact is added to the ChromaDB collection AT THE TURN it is introduced —
identical to how MDA receives knowledge. This is the fair comparison:
both systems receive the same information at the same time.

This directly addresses the reviewer concern:
  v1: RAG used a sliding context window (structurally weaker baseline)
  v2: RAG uses ChromaDB + incremental add() (production-grade baseline)

The structural gap being tested: even with incremental indexing, RAG
cannot learn mid-conversation nuance, cross-fact relationships, or
entity-level associations that MDA builds through synapse traversal.

Checkpoints: turn 10 | turn 50 | turn 200

Usage:
    # Ollama
    python long_context_benchmark_v2.py --model qwen3:4b

    # Anthropic
    python long_context_benchmark_v2.py --model claude-haiku-4-5-20251001 --provider anthropic

    # Compare both RAG variants
    python long_context_benchmark_v2.py --model qwen3:4b --rag-variant both
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer

import dotenv
dotenv.load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mda.integrations.engine import MDAEngine, AnthropicEngine

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    class _FallbackConsole:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("─" * 70)
    console = _FallbackConsole()


# ── World facts (identical to v1) ─────────────────────────────────────────────

CLUSTER_A: list[dict] = [
    {"id": "A1",  "text": "Solaris Research Station was founded in 2041 by Dr. Mira Voss.",
     "keywords": ["Mira Voss", "2041"]},
    {"id": "A2",  "text": "Solaris Station is located at coordinates 78.3°N, 22.1°E on the Arctic shelf.",
     "keywords": ["78.3°N", "22.1°E", "Arctic"]},
    {"id": "A3",  "text": "Solaris has 47 permanent staff and can host up to 120 visiting researchers.",
     "keywords": ["47", "120"]},
    {"id": "A4",  "text": "The station's primary power source is a compact fusion reactor named Helios-7.",
     "keywords": ["Helios-7", "fusion"]},
    {"id": "A5",  "text": "Dr. Mira Voss developed the Solaris Protocol, a zero-waste system that recycles 99.8 percent of all water.",
     "keywords": ["Solaris Protocol", "99.8"]},
    {"id": "A6",  "text": "Solaris Station's annual budget is 340 million credits, funded by the Global Science Consortium.",
     "keywords": ["340 million", "Global Science Consortium"]},
    {"id": "A7",  "text": "Solaris has a sub-ice ocean lab called Depth Chamber 3 at 800 meters below sea level.",
     "keywords": ["Depth Chamber 3", "800 meters"]},
    {"id": "A8",  "text": "Solaris Station's emergency protocol is called Code Amber, triggered when reactor output drops below 40 percent.",
     "keywords": ["Code Amber", "40"]},
    {"id": "A9",  "text": "Solaris discovered a new extremophile species named Cryobacter vossii in 2048.",
     "keywords": ["Cryobacter vossii", "2048"]},
    {"id": "A10", "text": "Solaris communicates via a dedicated satellite called POLARIS-1 in geostationary orbit.",
     "keywords": ["POLARIS-1", "geostationary"]},
]

CLUSTER_B: list[dict] = [
    {"id": "B1",  "text": "The Veridian Trade Network was established by the Merchant Council in 2055.",
     "keywords": ["Veridian Trade Network", "2055"]},
    {"id": "B2",  "text": "Veridian has 23 member nations and processes 4.7 trillion credits in annual trade.",
     "keywords": ["23", "4.7 trillion"]},
    {"id": "B3",  "text": "The Veridian currency is the Veri, pegged at 1 Veri equals 0.82 global credits.",
     "keywords": ["Veri", "0.82"]},
    {"id": "B4",  "text": "Veridian's headquarters is in the city of Aurum on the Meridian Coast.",
     "keywords": ["Aurum", "Meridian Coast"]},
    {"id": "B5",  "text": "Veridian operates the Fast-Track shipping system, reducing delivery time by 68 percent.",
     "keywords": ["Fast-Track", "68"]},
    {"id": "B6",  "text": "Veridian's director is Chancellor Opal Fenwick, elected in 2059.",
     "keywords": ["Opal Fenwick", "2059"]},
    {"id": "B7",  "text": "The Merchant Council meets quarterly in the Crystal Forum building in Aurum.",
     "keywords": ["Crystal Forum", "quarterly"]},
    {"id": "B8",  "text": "Veridian's anti-fraud system uses a blockchain called the Ledger Chain, audited every 72 hours.",
     "keywords": ["Ledger Chain", "72 hours"]},
    {"id": "B9",  "text": "Veridian's largest trade partner is Solaris Research Station, accounting for 12 percent of exports.",
     "keywords": ["Solaris", "12"]},
    {"id": "B10", "text": "Veridian imposes a 3.5 percent tariff on all non-member imports above 50,000 Veri.",
     "keywords": ["3.5", "50,000 Veri", "tariff"]},
]

CLUSTER_C: list[dict] = [
    {"id": "C1",  "text": "The Nexus AI system was developed by the Institute of Cognitive Machines in 2063.",
     "keywords": ["Nexus AI", "2063"]},
    {"id": "C2",  "text": "Nexus runs on a neuromorphic chip architecture called SynapseTech v4.",
     "keywords": ["SynapseTech v4", "neuromorphic"]},
    {"id": "C3",  "text": "Nexus AI has a processing speed of 10 to the power of 18 operations per second.",
     "keywords": ["10^18", "operations"]},
    {"id": "C4",  "text": "Nexus was deployed to manage Veridian Trade Network logistics in 2064.",
     "keywords": ["2064", "Veridian", "logistics"]},
    {"id": "C5",  "text": "Nexus learns via reinforcement from human feedback and updates its weights every 6 hours.",
     "keywords": ["reinforcement", "6 hours"]},
    {"id": "C6",  "text": "Nexus AI's chief architect is Dr. Yuri Okonkwo, formerly of Solaris Research Station.",
     "keywords": ["Yuri Okonkwo", "Solaris"]},
    {"id": "C7",  "text": "Nexus has a Guardian Mode self-preservation protocol that activates when power drops below 20 percent.",
     "keywords": ["Guardian Mode", "20"]},
    {"id": "C8",  "text": "Nexus monitors 1.2 billion transactions per day across the Veridian network.",
     "keywords": ["1.2 billion", "transactions"]},
    {"id": "C9",  "text": "Nexus operates from three redundant data centers: NX-Alpha, NX-Beta, and NX-Gamma.",
     "keywords": ["NX-Alpha", "NX-Beta", "NX-Gamma"]},
    {"id": "C10", "text": "In 2067 Nexus AI was accused of market manipulation and investigated by the Merchant Council.",
     "keywords": ["2067", "market manipulation", "Merchant Council"]},
]

TEST_CASES: list[dict] = [
    {"id": "A-T1", "cluster": "A-early", "cross": False,
     "q": "Who founded Solaris Research Station and in what year?",
     "kw": ["Mira Voss", "2041"],
     "ans": "Solaris Research Station was founded by Dr. Mira Voss in 2041."},
    {"id": "A-T2", "cluster": "A-early", "cross": False,
     "q": "What is the name of the reactor that powers Solaris Station?",
     "kw": ["Helios-7"],
     "ans": "Solaris Station is powered by a compact fusion reactor named Helios-7."},
    {"id": "A-T3", "cluster": "A-early", "cross": False,
     "q": "What emergency protocol does Solaris use and what threshold triggers it?",
     "kw": ["Code Amber", "40"],
     "ans": "Solaris uses Code Amber, triggered when reactor output drops below 40 percent."},
    {"id": "A-T4", "cluster": "A-early", "cross": False,
     "q": "What extremophile species did Solaris discover and when?",
     "kw": ["Cryobacter vossii", "2048"],
     "ans": "Solaris discovered Cryobacter vossii in 2048."},
    {"id": "A-T5", "cluster": "A-early", "cross": False,
     "q": "What is the Solaris Protocol and what percentage of water does it recycle?",
     "kw": ["Solaris Protocol", "99.8"],
     "ans": "The Solaris Protocol is a zero-waste system that recycles 99.8 percent of all water."},
    {"id": "B-T1", "cluster": "B-mid", "cross": False,
     "q": "Who is the director of the Veridian Trade Network and when were they elected?",
     "kw": ["Opal Fenwick", "2059"],
     "ans": "Veridian's director is Chancellor Opal Fenwick, elected in 2059."},
    {"id": "B-T2", "cluster": "B-mid", "cross": False,
     "q": "What is the Veri exchange rate to global credits?",
     "kw": ["0.82"],
     "ans": "1 Veri equals 0.82 global credits."},
    {"id": "B-T3", "cluster": "B-mid", "cross": False,
     "q": "What is the Ledger Chain and how often is it audited?",
     "kw": ["Ledger Chain", "72"],
     "ans": "The Ledger Chain is Veridian's anti-fraud blockchain, audited every 72 hours."},
    {"id": "B-T4", "cluster": "B-mid", "cross": False,
     "q": "How much does Veridian reduce shipping time with its Fast-Track system?",
     "kw": ["68"],
     "ans": "Veridian's Fast-Track system reduces shipping time by 68 percent."},
    {"id": "B-T5", "cluster": "B-mid", "cross": False,
     "q": "What tariff does Veridian impose on non-member imports?",
     "kw": ["3.5", "50,000"],
     "ans": "Veridian imposes a 3.5 percent tariff on non-member imports above 50,000 Veri."},
    {"id": "C-T1", "cluster": "C-late", "cross": False,
     "q": "What chip architecture does Nexus AI run on?",
     "kw": ["SynapseTech v4", "neuromorphic"],
     "ans": "Nexus AI runs on a neuromorphic chip architecture called SynapseTech v4."},
    {"id": "C-T2", "cluster": "C-late", "cross": False,
     "q": "What self-preservation mode does Nexus AI have and what activates it?",
     "kw": ["Guardian Mode", "20"],
     "ans": "Nexus AI has Guardian Mode, which activates when power drops below 20 percent."},
    {"id": "C-T3", "cluster": "C-late", "cross": False,
     "q": "How many transactions per day does Nexus monitor?",
     "kw": ["1.2 billion"],
     "ans": "Nexus monitors 1.2 billion transactions per day."},
    {"id": "C-T4", "cluster": "C-late", "cross": False,
     "q": "What are the three data centers where Nexus operates?",
     "kw": ["NX-Alpha", "NX-Beta", "NX-Gamma"],
     "ans": "Nexus operates from NX-Alpha, NX-Beta, and NX-Gamma."},
    {"id": "C-T5", "cluster": "C-late", "cross": False,
     "q": "What happened to Nexus AI in 2067?",
     "kw": ["market manipulation", "Merchant Council"],
     "ans": "Nexus AI was accused of market manipulation and investigated by the Merchant Council in 2067."},
    {"id": "X-T1", "cluster": "cross", "cross": True,
     "q": "Who is Nexus AI's chief architect and what is his connection to Solaris?",
     "kw": ["Yuri Okonkwo", "Solaris"],
     "ans": "Dr. Yuri Okonkwo is Nexus AI's chief architect and was formerly of Solaris Research Station."},
    {"id": "X-T2", "cluster": "cross", "cross": True,
     "q": "What percentage of Veridian exports go to Solaris, and who founded Solaris?",
     "kw": ["12", "Mira Voss"],
     "ans": "12 percent of Veridian exports go to Solaris Research Station, which was founded by Dr. Mira Voss."},
    {"id": "X-T3", "cluster": "cross", "cross": True,
     "q": "When was Nexus AI deployed to manage Veridian logistics, and who is Veridian's director?",
     "kw": ["2064", "Opal Fenwick"],
     "ans": "Nexus was deployed to Veridian logistics in 2064; Veridian's director is Chancellor Opal Fenwick."},
    {"id": "X-T4", "cluster": "cross", "cross": True,
     "q": "What body investigated Nexus AI for market manipulation, and where does that body meet?",
     "kw": ["Merchant Council", "Crystal Forum"],
     "ans": "The Merchant Council investigated Nexus AI; it meets in the Crystal Forum in Aurum."},
    {"id": "X-T5", "cluster": "cross", "cross": True,
     "q": "What are the emergency power thresholds for Solaris Station and Nexus AI respectively?",
     "kw": ["40", "20"],
     "ans": "Solaris triggers Code Amber at 40 percent reactor output; Nexus activates Guardian Mode at 20 percent power."},
]

FILLER_MESSAGES = [
    "What is the capital of France?",
    "Can you write a haiku about rain?",
    "How does photosynthesis work?",
    "What is the speed of light in a vacuum?",
    "Explain the concept of entropy briefly.",
    "How many bones are in the human body?",
    "What causes the Northern Lights?",
    "Tell me one fact about ocean tides.",
    "What is the difference between a comet and an asteroid?",
    "How does a vaccine work?",
    "What is the largest planet in the solar system?",
    "Briefly explain what DNA is.",
    "What is the Pythagorean theorem?",
    "How does bread rise when baked?",
    "What is the difference between weather and climate?",
    "What is sonar and how does it work?",
    "How do birds navigate during migration?",
    "What is the approximate age of the universe?",
    "What causes earthquakes?",
    "How does a compass work?",
]


def build_conversation_script() -> list[dict]:
    script: list[dict] = []

    def teach(fact: dict, phase: str) -> dict:
        return {
            "user": f"Please remember this fact: {fact['text']} Acknowledge by repeating the key detail.",
            "fact_id": fact["id"],
            "fact_text": fact["text"],
            "phase": phase,
        }

    def filler(idx: int) -> dict:
        return {
            "user": FILLER_MESSAGES[idx % len(FILLER_MESSAGES)],
            "fact_id": None,
            "fact_text": None,
            "phase": "filler",
        }

    for fact in CLUSTER_A:
        script.append(teach(fact, "A"))
    for i in range(14):
        script.append(filler(i))
    b_idx = 0; f_idx = 14
    for _ in range(20):
        if b_idx < len(CLUSTER_B):
            script.append(teach(CLUSTER_B[b_idx], "B")); b_idx += 1
        else:
            script.append(filler(f_idx)); f_idx += 1
    for i in range(55):
        script.append(filler(f_idx + i))
    f_idx += 55
    c_idx = 0
    for _ in range(20):
        if c_idx < len(CLUSTER_C):
            script.append(teach(CLUSTER_C[c_idx], "C")); c_idx += 1
        else:
            script.append(filler(f_idx)); f_idx += 1
    for i in range(81):
        script.append(filler(f_idx + i))

    assert len(script) == 200, f"Expected 200 turns, got {len(script)}"
    return script


# ── LLM router ─────────────────────────────────────────────────────────────────

def _llm(model: str, messages: list[dict], provider: str, api_key: str = "", base_url: str = "") -> str:
    if provider == "anthropic":
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=256, system=system, messages=user_msgs,
        )
        raw = resp.content[0].text
    elif provider == "llama_cpp":
        import requests
        url = base_url or "http://localhost:11435/v1/chat/completions"
        resp = requests.post(url, json={
            "model": model,
            "messages": messages,
            "max_tokens": 256,
        }, timeout=120)
        raw = resp.json()["choices"][0]["message"]["content"]
    else:
        import ollama as _ollama
        raw = _ollama.chat(model=model, messages=messages)["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── Incremental ChromaDB RAG ────────────────────────────────────────────────────

class IncrementalChromaRAG:
    """
    Production-grade RAG baseline with incremental indexing.

    Every fact is added to ChromaDB AT THE TURN it is introduced —
    identical to MDA's learn() call timing. Uses ChromaDB's default
    embedding function (all-MiniLM-L6-v2 via sentence-transformers,
    or falls back to ChromaDB's built-in embeddings).

    This is the FAIR comparison baseline:
      - Same information, same timing as MDA
      - Persistent vector index (no sliding window)
      - top-6 retrieval (matches main benchmark config)
    """

    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str = "",
        base_url: str = "",
        top_k: int = 6,
        collection_name: str = "lcb_bench",
    ) -> None:
        if not HAS_CHROMA:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.model      = model
        self.provider   = provider
        self.api_key    = api_key
        self.base_url   = base_url
        self.top_k      = top_k
        self._doc_count = 0

        # In-memory ChromaDB client — no disk persistence needed for benchmark
        self._client = chromadb.Client()

        # MDA HolisticEncoder used for embeddings — no extra dependencies,
        # same encoder used by MDA itself. For bge-large-en-v1.5 (matches main
        # benchmark), install sentence-transformers and swap self._encoder below.
        from mda.core.encoder import HolisticEncoder
        from mda.core.bind import DIM
        self._encoder = HolisticEncoder(DIM)

        # embedding_function=None: we supply embeddings manually
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not hasattr(self, '_st_model'):
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        return self._st_model.encode(texts, normalize_embeddings=True).tolist()

    def add_fact(self, fact_id: str, fact_text: str) -> None:
        """Add a fact to the index — called at the same turn as mda.learn()."""
        self._collection.add(
            documents=[fact_text],
            embeddings=self._embed([fact_text]),
            ids=[fact_id],
        )
        self._doc_count += 1

    def retrieve(self, query: str) -> str:
        """Retrieve top-k relevant facts and return as context string."""
        if self._doc_count == 0:
            return ""
        k = min(self.top_k, self._doc_count)
        results = self._collection.query(
            query_embeddings=self._embed([query]),
            n_results=k,
        )
        docs = results["documents"][0] if results["documents"] else []
        return "\n".join(docs)

    def query(self, question: str) -> str:
        """Retrieve context then ask LLM."""
        context = self.retrieve(question)
        if not context:
            system = "You are a helpful assistant. Answer based on what you know."
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ]
        else:
            system = (
                "You are a helpful assistant. Answer the question using ONLY the "
                "provided context. If the answer is not in the context, say you don't know. "
                "Be concise and precise.\n\n"
                f"Context:\n{context}"
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ]
        return _llm(self.model, messages, self.provider, self.api_key, base_url=self.base_url)

    @property
    def indexed_facts(self) -> int:
        return self._doc_count


# ── Sliding window RAG (v1 baseline, kept for comparison) ──────────────────────

class SlidingWindowRAG:
    """Original v1 baseline — kept for direct comparison with v2."""

    def __init__(self, model, provider, api_key="", base_url="", max_context_chars=15_000):
        self.model = model; self.provider = provider; self.api_key = api_key
        self.base_url = base_url
        self.max_context_chars = max_context_chars
        self.history: list[dict] = []

    def _system(self):
        return "You are a helpful assistant. Answer based on what has been discussed. Be concise."

    def _trimmed(self, extra=0):
        budget = self.max_context_chars - extra
        result = []; chars = 0
        for msg in reversed(self.history):
            c = len(msg["content"])
            if chars + c > budget: break
            result.insert(0, msg); chars += c
        return result

    def chat(self, user_msg):
        self.history.append({"role": "user", "content": user_msg})
        msgs = [{"role": "system", "content": self._system()}] + self._trimmed(len(user_msg))
        resp = _llm(self.model, msgs, self.provider, self.api_key, base_url=self.base_url)
        self.history.append({"role": "assistant", "content": resp})
        return resp

    def query(self, question):
        msgs = [{"role": "system", "content": self._system()}] + self._trimmed(len(question))
        msgs.append({"role": "user", "content": question})
        return _llm(self.model, msgs, self.provider, self.api_key, base_url=self.base_url)


# ── MDA wrapper ─────────────────────────────────────────────────────────────────

def build_mda_engine(model: str, provider: str, api_key: str, base_url: str = "") -> MDAEngine:
    if provider == "anthropic":
        return AnthropicEngine(model=model, user_id="lcb_v2_bench", api_key=api_key)
    return MDAEngine(model=model, user_id="lcb_v2_bench",
                     provider=provider, base_url=base_url)


# ── Judge ────────────────────────────────────────────────────────────────────────

def judge(question, answer, expected, keywords, model, provider, api_key="", base_url=""):
    kw_hits  = sum(1 for kw in keywords if kw.lower() in answer.lower())
    kw_score = round(kw_hits / len(keywords), 2) if keywords else 0.0
    prompt = (
        "Score this answer vs the expected answer.\n"
        "2 = fully correct  1 = partially correct  0 = wrong or unknown\n\n"
        f"Question: {question}\nExpected: {expected}\nAnswer: {answer}\n\n"
        "Reply with a single digit: 0, 1, or 2."
    )
    try:
        raw   = _llm(model, [{"role": "user", "content": prompt}], provider, api_key, base_url=base_url)
        match = re.search(r"[012]", raw)
        score = int(match.group(0)) if match else 0
    except Exception as e:
        score = 0
        console.print(f"[red]  judge error: {e}[/red]")
    return {"score": score, "kw_score": kw_score, "kw_hits": kw_hits, "kw_total": len(keywords)}


# ── Core benchmark loop ─────────────────────────────────────────────────────────

def run_turns(
    mda: MDAEngine,
    rag_chroma: IncrementalChromaRAG,
    rag_sliding: SlidingWindowRAG | None,
    script: list[dict],
    start: int,
    end: int,
) -> None:
    for i in range(start, end):
        turn = script[i]
        user_msg   = turn["user"]
        fact_text  = turn["fact_text"]
        fact_id    = turn["fact_id"]

        # MDA: chat + explicit learn on fact turns
        mda.chat(user_msg, lang="en")
        if fact_id is not None:
            mda.learn(user_msg)

        # ChromaDB RAG: add fact to index immediately
        if fact_id is not None and fact_text is not None:
            rag_chroma.add_fact(fact_id, fact_text)

        # Sliding window RAG (optional v1 comparison)
        if rag_sliding is not None:
            rag_sliding.chat(user_msg)

        if (i + 1) % 10 == 0:
            console.print(
                f"[dim]  turn {i+1:3d} | "
                f"ChromaDB indexed: {rag_chroma.indexed_facts} facts | "
                f"MDA entities: {len(mda.mda.registry._entities)}[/dim]"
            )


def measure_accuracy(
    mda: MDAEngine,
    rag_chroma: IncrementalChromaRAG,
    rag_sliding: SlidingWindowRAG | None,
    test_cases: list[dict],
    judge_model: str,
    judge_provider: str,
    api_key: str,
    checkpoint: int,
    judge_base_url: str = "",
) -> list[dict]:
    results = []
    for tc in test_cases:
        chroma_ans  = rag_chroma.query(tc["q"])
        mda_ans     = mda.query(tc["q"], lang="en")
        sliding_ans = rag_sliding.query(tc["q"]) if rag_sliding else None

        chroma_j  = judge(tc["q"], chroma_ans,  tc["ans"], tc["kw"], judge_model, judge_provider, api_key, base_url=judge_base_url)
        mda_j     = judge(tc["q"], mda_ans,     tc["ans"], tc["kw"], judge_model, judge_provider, api_key, base_url=judge_base_url)
        sliding_j = judge(tc["q"], sliding_ans, tc["ans"], tc["kw"], judge_model, judge_provider, api_key, base_url=judge_base_url) if sliding_ans else None

        row = {
            "checkpoint": checkpoint,
            "id":         tc["id"],
            "cluster":    tc["cluster"],
            "cross":      tc["cross"],
            "question":   tc["q"],
            "chroma_ans": chroma_ans,
            "mda_ans":    mda_ans,
            "chroma":     chroma_j,
            "mda":        mda_j,
        }
        if sliding_j:
            row["sliding_ans"] = sliding_ans
            row["sliding"]     = sliding_j

        results.append(row)

        cs = chroma_j["score"]; ms = mda_j["score"]
        tag = "TIE" if cs == ms else ("MDA↑" if ms > cs else "RAG↑")
        console.print(
            f"  [dim][{tc['id']}][/dim] ChromaRAG={cs} MDA={ms} {tag}"
            + (f" SlidingRAG={sliding_j['score']}" if sliding_j else "")
            + f" | {tc['q'][:50]}"
        )
    return results


# ── Reporting ────────────────────────────────────────────────────────────────────

def _pct(score, n):
    if n == 0: return "—"
    return f"{score / (n * 2) * 100:.0f}%"


def print_checkpoint_summary(results: list[dict], checkpoint: int) -> None:
    console.rule(f"[bold]Checkpoint — Turn {checkpoint}[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Cluster",      width=12)
    table.add_column("N",            width=4,  justify="right")
    table.add_column("ChromaRAG",    width=11, justify="center")
    table.add_column("MDA",          width=8,  justify="center")
    table.add_column("Winner",       width=10, justify="center")

    has_sliding = any("sliding" in r for r in results)
    if has_sliding:
        table.add_column("SlidingRAG", width=11, justify="center")

    by_cluster: dict[str, dict] = {}
    for r in results:
        c = r["cluster"]
        if c not in by_cluster:
            by_cluster[c] = {"chroma": 0, "mda": 0, "sliding": 0, "n": 0}
        by_cluster[c]["chroma"]  += r["chroma"]["score"]
        by_cluster[c]["mda"]     += r["mda"]["score"]
        by_cluster[c]["sliding"] += r.get("sliding", {}).get("score", 0)
        by_cluster[c]["n"]       += 1

    for cluster, s in sorted(by_cluster.items()):
        cs, ms, n = s["chroma"], s["mda"], s["n"]
        row = [
            cluster, str(n), _pct(cs, n), _pct(ms, n),
            "[green]MDA[/green]" if ms > cs else ("[red]ChromaRAG[/red]" if cs > ms else "[dim]TIE[/dim]"),
        ]
        if has_sliding:
            row.append(_pct(s["sliding"], n))
        table.add_row(*row)

    tc = sum(s["chroma"] for s in by_cluster.values())
    tm = sum(s["mda"]    for s in by_cluster.values())
    nt = sum(s["n"]      for s in by_cluster.values())
    total_row = [
        "[bold]TOTAL[/bold]", str(nt),
        f"[bold]{_pct(tc, nt)}[/bold]",
        f"[bold]{_pct(tm, nt)}[/bold]",
        "[bold green]MDA[/bold green]" if tm > tc else ("[bold red]ChromaRAG[/bold red]" if tc > tm else "[bold]TIE[/bold]"),
    ]
    if has_sliding:
        ts = sum(s["sliding"] for s in by_cluster.values())
        total_row.append(f"[bold]{_pct(ts, nt)}[/bold]")
    table.add_row(*total_row)
    console.print(table)


def print_final_report(all_results: dict, checkpoints: list[int]) -> None:
    console.rule("[bold]Final: Incremental ChromaRAG vs MDA[/bold]")
    clusters = ["A-early", "B-mid", "C-late", "cross"]

    table = Table(
        show_header=True, header_style="bold cyan",
        title="Accuracy % — Incremental ChromaRAG vs MDA by Cluster × Checkpoint"
    )
    table.add_column("Cluster", width=14)
    for cp in checkpoints:
        table.add_column(f"T={cp} ChromaRAG", width=14, justify="center")
        table.add_column(f"T={cp} MDA",       width=10, justify="center")

    for cluster in clusters:
        row = [cluster]
        for cp in checkpoints:
            rs = all_results.get(cp, [])
            sub = [r for r in rs if r["cluster"] == cluster]
            n   = len(sub)
            cs  = sum(r["chroma"]["score"] for r in sub)
            ms  = sum(r["mda"]["score"]    for r in sub)
            row += [_pct(cs, n), _pct(ms, n)]
        table.add_row(*row)

    row = ["[bold]TOTAL[/bold]"]
    for cp in checkpoints:
        rs = all_results.get(cp, [])
        n  = len(rs)
        cs = sum(r["chroma"]["score"] for r in rs)
        ms = sum(r["mda"]["score"]    for r in rs)
        row += [f"[bold]{_pct(cs, n)}[/bold]", f"[bold]{_pct(ms, n)}[/bold]"]
    table.add_row(*row)
    console.print(table)

    # Narrative
    console.print("\n[bold]Key observations:[/bold]")
    cluster_labels = {
        "A-early": "A-early (turns  1-10)",
        "B-mid":   "B-mid   (turns 25-34)",
        "C-late":  "C-late  (turns 100-109)",
        "cross":   "Cross-cluster",
    }
    for cluster in clusters:
        first_cp = checkpoints[0]; last_cp = checkpoints[-1]
        r0 = [r for r in all_results.get(first_cp, []) if r["cluster"] == cluster]
        r1 = [r for r in all_results.get(last_cp,  []) if r["cluster"] == cluster]
        if not r0 or not r1: continue
        c0 = sum(r["chroma"]["score"] for r in r0) / (len(r0) * 2) * 100
        c1 = sum(r["chroma"]["score"] for r in r1) / (len(r1) * 2) * 100
        m0 = sum(r["mda"]["score"]    for r in r0) / (len(r0) * 2) * 100
        m1 = sum(r["mda"]["score"]    for r in r1) / (len(r1) * 2) * 100
        cc = "red" if c1 - c0 < -5 else ("green" if c1 - c0 > 5 else "dim")
        mc = "green" if m1 - m0 > -5 else "red"
        console.print(
            f"  {cluster_labels[cluster]}: "
            f"ChromaRAG [{cc}]{c0:.0f}%→{c1:.0f}% ({c1-c0:+.0f})[/{cc}]  "
            f"MDA [{mc}]{m0:.0f}%→{m1:.0f}% ({m1-m0:+.0f})[/{mc}]"
        )


def save_results(all_results: dict, output_dir: str, model: str) -> None:
    out  = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace(":", "-").replace("/", "-")
    path = out / f"lcb_v2_chroma_{slug}_{ts}.json"
    path.write_text(
        json.dumps({str(k): v for k, v in all_results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Results saved → {path}[/dim]")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-Context Benchmark v2: MDA vs Incremental ChromaDB RAG"
    )
    parser.add_argument("--model",          default="qwen3:4b")
    parser.add_argument("--provider",       default="ollama", choices=["ollama", "anthropic", "llama_cpp"])
    parser.add_argument("--api-key",        default="")
    parser.add_argument("--base-url",       default="",
                        help="llama.cpp server URL (default: http://localhost:11435/v1/chat/completions)")
    parser.add_argument("--judge",          default=None)
    parser.add_argument("--judge-provider", default=None, choices=["ollama", "anthropic", "llama_cpp"])
    parser.add_argument("--judge-base-url", default="",
                        help="llama.cpp judge server URL, falls back to --base-url if empty")
    parser.add_argument("--top-k",          type=int, default=6,
                        help="ChromaDB retrieval top-k (default: 6, matches main benchmark)")
    parser.add_argument("--rag-variant",    default="chroma",
                        choices=["chroma", "sliding", "both"],
                        help="Which RAG baseline to run alongside MDA")
    parser.add_argument("--sliding-window", type=int, default=15_000,
                        help="Sliding window size if --rag-variant=both (default: 15000)")
    parser.add_argument("--output",         default="results")
    parser.add_argument("--checkpoints",    nargs="+", type=int, default=[10, 50, 200])
    args = parser.parse_args()

    if not HAS_CHROMA:
        console.print("[red]chromadb not installed. Run: pip install chromadb[/red]")
        sys.exit(1)

    checkpoints   = sorted(args.checkpoints)
    judge_model   = args.judge    or args.model
    judge_prov    = args.judge_provider or args.provider
    api_key       = args.api_key
    use_sliding   = args.rag_variant in ("sliding", "both")

    console.rule("[bold]Long-Context Benchmark v2: MDA vs Incremental ChromaDB RAG[/bold]")
    console.print(f"[dim]  model:       {args.model}  provider: {args.provider}[/dim]")
    console.print(f"[dim]  judge:       {judge_model}  judge-provider: {judge_prov}[/dim]")
    console.print(f"[dim]  RAG variant: {args.rag_variant}  ChromaDB top-k: {args.top_k}[/dim]")
    console.print(f"[dim]  checkpoints: {checkpoints}[/dim]\n")
    console.print("[yellow]NOTE: ChromaDB RAG receives identical information at identical turns as MDA.[/yellow]")
    console.print("[yellow]      This is the fair incremental comparison (v1 used sliding window).[/yellow]\n")

    script = build_conversation_script()

    console.print("[bold yellow]Initialising MDA engine...[/bold yellow]")
    mda = build_mda_engine(args.model, args.provider, api_key, base_url=args.base_url)

    console.print("[bold yellow]Initialising Incremental ChromaDB RAG...[/bold yellow]")
    rag_chroma = IncrementalChromaRAG(
        model=args.model, provider=args.provider, api_key=api_key,
        base_url=args.base_url, top_k=args.top_k,
    )

    rag_sliding = None
    if use_sliding:
        console.print("[bold yellow]Initialising Sliding Window RAG (v1 baseline)...[/bold yellow]")
        rag_sliding = SlidingWindowRAG(
            model=args.model, provider=args.provider, api_key=api_key,
            base_url=args.base_url, max_context_chars=args.sliding_window,
        )

    all_results: dict[int, list[dict]] = {}
    last_turn = 0

    for cp in checkpoints:
        if cp > 200:
            console.print(f"[yellow]Checkpoint {cp} > 200 turns — skipping.[/yellow]")
            continue
        if cp > last_turn:
            console.print(f"\n[bold yellow]Running turns {last_turn + 1}–{cp}...[/bold yellow]")
            t0 = time.time()
            run_turns(mda, rag_chroma, rag_sliding, script, last_turn, cp)
            console.print(f"[dim]  Done in {time.time()-t0:.1f}s[/dim]")
            last_turn = cp

        console.print(f"\n[bold yellow]Measuring accuracy at turn {cp}...[/bold yellow]")
        console.print(
            f"[dim]  ChromaDB: {rag_chroma.indexed_facts} facts indexed | "
            f"MDA entities: {len(mda.mda.registry._entities)}[/dim]"
        )
        t0 = time.time()
        results = measure_accuracy(
            mda, rag_chroma, rag_sliding, TEST_CASES,
            judge_model, judge_prov, api_key, cp,
            judge_base_url=args.judge_base_url or args.base_url,
        )
        console.print(f"[dim]  Done in {time.time()-t0:.1f}s[/dim]")
        all_results[cp] = results
        print_checkpoint_summary(results, cp)

    print_final_report(all_results, checkpoints)
    save_results(all_results, args.output, args.model)


if __name__ == "__main__":
    main()