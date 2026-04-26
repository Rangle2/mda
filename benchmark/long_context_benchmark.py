"""
Long-Context Accuracy Benchmark: MDA vs RAG
============================================
Tests how accuracy evolves across 10 → 50 → 200 conversation turns.

Core hypothesis:
  RAG  — history grows, oldest facts fall outside the context window;
          accuracy on early facts collapses by turn 200.
  MDA  — entity graph and synapse strengths accumulate; accuracy on
          early facts stays stable or improves.

Checkpoints: turn 10 | turn 50 | turn 200

How it works:
  1. A synthetic "world" has 30 facts in 3 clusters (A/B/C).
  2. A 200-turn conversation introduces them gradually:
       Turns   1-10  → Cluster A (Solaris Research Station)
       Turns  25-44  → Cluster B (Veridian Trade Network)
       Turns 100-119 → Cluster C (Nexus AI)
       Remaining turns are off-topic filler that push facts out of RAG's window.
  3. At each checkpoint, 20 test questions are fired at both systems.
  4. A judge LLM scores every answer 0–2.
  5. Scores are broken down by fact age (early / middle / late) and cluster.

Usage:
    # Anthropic
    python long_context_benchmark.py --model claude-haiku-4-5-20251001 --provider anthropic

    # Ollama
    python long_context_benchmark.py --model qwen3:4b

    # Smaller RAG window to amplify the effect
    python long_context_benchmark.py --model qwen3:4b --rag-window 6000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
dotenv.load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from integrations.engine import MDAEngine, AnthropicEngine

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    class _FallbackConsole:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("─" * 70)
    console = _FallbackConsole()


# ── Synthetic world: 30 facts in 3 clusters ────────────────────────────────────

CLUSTER_A: list[dict] = [
    {"id": "A1",  "text": "Solaris Research Station was founded in 2041 by Dr. Mira Voss.",
     "keywords": ["2041", "Mira Voss"]},
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

# ── Fixed test set (evaluated at every checkpoint) ─────────────────────────────
# 5 per cluster (early / middle / late) + 5 cross-cluster

TEST_CASES: list[dict] = [
    # ── Cluster A ─────────────────────────────────────────────────────────────
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

    # ── Cluster B ─────────────────────────────────────────────────────────────
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

    # ── Cluster C ─────────────────────────────────────────────────────────────
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

    # ── Cross-cluster (require combining facts from 2+ clusters) ──────────────
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


# ── Conversation script builder ────────────────────────────────────────────────

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
    """
    Returns 200 turns. Each turn is a dict:
      { "user": str, "fact_id": str | None, "phase": str }

    Facts are introduced via explicit "remember this fact" messages so both
    MDA and RAG receive the information in the same form.
    Filler messages are off-topic questions that consume RAG's context window.
    """
    script: list[dict] = []

    def teach(fact: dict, phase: str) -> dict:
        return {
            "user": (
                f"Please remember this fact: {fact['text']} "
                f"Acknowledge by repeating the key detail."
            ),
            "fact_id": fact["id"],
            "phase": phase,
        }

    def filler(idx: int) -> dict:
        return {
            "user": FILLER_MESSAGES[idx % len(FILLER_MESSAGES)],
            "fact_id": None,
            "phase": "filler",
        }

    # Turns 1-10: Cluster A facts
    for fact in CLUSTER_A:
        script.append(teach(fact, "A"))

    # Turns 11-24: filler
    for i in range(14):
        script.append(filler(i))

    # Turns 25-44: Cluster B facts (10) + 10 filler interleaved
    b_idx = 0
    f_idx = 14
    for _ in range(20):
        if b_idx < len(CLUSTER_B):
            script.append(teach(CLUSTER_B[b_idx], "B"))
            b_idx += 1
        else:
            script.append(filler(f_idx))
            f_idx += 1

    # Turns 45-99: filler (55 turns)
    for i in range(55):
        script.append(filler(f_idx + i))
    f_idx += 55

    # Turns 100-119: Cluster C facts (10) + 10 filler interleaved
    c_idx = 0
    for _ in range(20):
        if c_idx < len(CLUSTER_C):
            script.append(teach(CLUSTER_C[c_idx], "C"))
            c_idx += 1
        else:
            script.append(filler(f_idx))
            f_idx += 1

    # Turns 120-200: filler (81 turns) — pushes A and B facts out of RAG window
    for i in range(81):
        script.append(filler(f_idx + i))

    assert len(script) == 200, f"Expected 200 turns, got {len(script)}"
    return script


# ── LLM router ─────────────────────────────────────────────────────────────────

def _llm(model: str, messages: list[dict], provider: str, api_key: str = "") -> str:
    if provider == "anthropic":
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=256, system=system, messages=user_msgs,
        )
        raw = resp.content[0].text
    else:
        import ollama as _ollama
        raw = _ollama.chat(model=model, messages=messages)["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── Naive RAG bot ───────────────────────────────────────────────────────────────

class NaiveRAGBot:
    """
    Simulates a RAG-style chatbot that keeps conversation history in a
    sliding context window. When history exceeds max_context_chars, the
    oldest messages are dropped — this is the failure mode under test.
    """

    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str = "",
        max_context_chars: int = 15_000,
    ) -> None:
        self.model            = model
        self.provider         = provider
        self.api_key          = api_key
        self.max_context_chars = max_context_chars
        self.history: list[dict] = []  # {"role": "user"|"assistant", "content": "..."}

    def _system(self) -> str:
        return (
            "You are a helpful assistant. Answer based on what has been "
            "discussed in this conversation. Be concise."
        )

    def _trimmed_history(self, extra_chars: int = 0) -> list[dict]:
        """Return history trimmed to fit within max_context_chars."""
        budget = self.max_context_chars - extra_chars
        result: list[dict] = []
        chars = 0
        for msg in reversed(self.history):
            c = len(msg["content"])
            if chars + c > budget:
                break
            result.insert(0, msg)
            chars += c
        return result

    def chat(self, user_message: str) -> str:
        """Process a conversation turn — adds to history."""
        self.history.append({"role": "user", "content": user_message})
        msgs = [{"role": "system", "content": self._system()}]
        msgs += self._trimmed_history(extra_chars=len(user_message))
        response = _llm(self.model, msgs, self.provider, self.api_key)
        self.history.append({"role": "assistant", "content": response})
        return response

    def query(self, question: str) -> str:
        """Query without modifying history (accuracy evaluation only)."""
        msgs = [{"role": "system", "content": self._system()}]
        msgs += self._trimmed_history(extra_chars=len(question))
        msgs.append({"role": "user", "content": question})
        return _llm(self.model, msgs, self.provider, self.api_key)

    @property
    def history_chars(self) -> int:
        return sum(len(m["content"]) for m in self.history)

    @property
    def visible_turns(self) -> int:
        """How many history messages are currently within the context window."""
        return len(self._trimmed_history())


# ── MDA wrapper ─────────────────────────────────────────────────────────────────

def build_mda_engine(model: str, provider: str, api_key: str) -> MDAEngine:
    if provider == "anthropic":
        return AnthropicEngine(model=model, user_id="lcb_bench", api_key=api_key)
    return MDAEngine(model=model, user_id="lcb_bench")


# ── Judge ────────────────────────────────────────────────────────────────────────

def judge(
    question: str,
    answer: str,
    expected_answer: str,
    keywords: list[str],
    model: str,
    provider: str,
    api_key: str = "",
) -> dict:
    """Score an answer 0–2 using keyword check + LLM judge."""
    kw_hits  = sum(1 for kw in keywords if kw.lower() in answer.lower())
    kw_score = round(kw_hits / len(keywords), 2) if keywords else 0.0

    prompt = (
        "Compare the answer below to the expected answer and assign a score.\n"
        "2 = fully correct\n"
        "1 = partially correct or missing key details\n"
        "0 = wrong, irrelevant, or says it doesn't know\n\n"
        f"Question: {question}\n"
        f"Expected: {expected_answer}\n"
        f"Answer:   {answer}\n\n"
        "Reply with a single digit: 0, 1, or 2. Nothing else."
    )
    try:
        raw   = _llm(model, [{"role": "user", "content": prompt}], provider, api_key)
        match = re.search(r"[012]", raw)
        score = int(match.group(0)) if match else 0
    except Exception as exc:
        score = 0
        console.print(f"[red]  judge error: {exc}[/red]")

    return {"score": score, "kw_score": kw_score, "kw_hits": kw_hits, "kw_total": len(keywords)}


# ── Core routines ────────────────────────────────────────────────────────────────

def run_turns(
    mda: MDAEngine,
    rag: NaiveRAGBot,
    script: list[dict],
    start: int,
    end: int,
) -> None:
    """Drive both systems through script[start:end] (0-indexed)."""
    for i in range(start, end):
        turn = script[i]
        user_msg = turn["user"]
        rag.chat(user_msg)
        mda.chat(user_msg, lang="en")
        if turn["fact_id"] is not None:
            mda.learn(user_msg)
        if (i + 1) % 10 == 0:
            console.print(
                f"[dim]  turn {i+1:3d} | RAG history: {rag.history_chars:6,} chars "
                f"({rag.visible_turns} visible) | MDA entities: "
                f"{len(mda.mda.registry._entities)}[/dim]"
            )


def measure_accuracy(
    mda: MDAEngine,
    rag: NaiveRAGBot,
    test_cases: list[dict],
    judge_model: str,
    judge_provider: str,
    api_key: str,
    checkpoint: int,
) -> list[dict]:
    """Run all test questions against both systems, return scored results."""
    results = []
    for tc in test_cases:
        rag_ans = rag.query(tc["q"])
        mda_ans = mda.query(tc["q"], lang="en")

        rag_j = judge(tc["q"], rag_ans, tc["ans"], tc["kw"],
                      judge_model, judge_provider, api_key)
        mda_j = judge(tc["q"], mda_ans, tc["ans"], tc["kw"],
                      judge_model, judge_provider, api_key)

        results.append({
            "checkpoint": checkpoint,
            "id":         tc["id"],
            "cluster":    tc["cluster"],
            "cross":      tc["cross"],
            "question":   tc["q"],
            "rag_ans":    rag_ans,
            "mda_ans":    mda_ans,
            "rag":        rag_j,
            "mda":        mda_j,
        })
        rs, ms = rag_j["score"], mda_j["score"]
        tag = "TIE" if rs == ms else ("MDA↑" if ms > rs else "RAG↑")
        console.print(
            f"  [dim][{tc['id']}][/dim] RAG={rs} MDA={ms} {tag} "
            f"| {tc['q'][:55]}"
        )
    return results


# ── Reporting ────────────────────────────────────────────────────────────────────

def _pct(score: float, n: int) -> str:
    if n == 0:
        return "—"
    return f"{score / (n * 2) * 100:.0f}%"


def print_checkpoint_summary(results: list[dict], checkpoint: int) -> None:
    console.rule(f"[bold]Checkpoint — Turn {checkpoint}[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Cluster", width=10)
    table.add_column("N", width=4, justify="right")
    table.add_column("RAG score", width=10, justify="center")
    table.add_column("MDA score", width=10, justify="center")
    table.add_column("Winner", width=8, justify="center")

    by_cluster: dict[str, dict] = {}
    for r in results:
        c = r["cluster"]
        if c not in by_cluster:
            by_cluster[c] = {"rag": 0, "mda": 0, "n": 0}
        by_cluster[c]["rag"] += r["rag"]["score"]
        by_cluster[c]["mda"] += r["mda"]["score"]
        by_cluster[c]["n"]   += 1

    for cluster, s in sorted(by_cluster.items()):
        rs, ms, n = s["rag"], s["mda"], s["n"]
        pct_r = _pct(rs, n)
        pct_m = _pct(ms, n)
        if ms > rs:
            winner = "[green]MDA[/green]"
        elif rs > ms:
            winner = "[red]RAG[/red]"
        else:
            winner = "[dim]TIE[/dim]"
        table.add_row(cluster, str(n), pct_r, pct_m, winner)

    # Totals
    total_r = sum(r["rag"]["score"] for r in results)
    total_m = sum(r["mda"]["score"] for r in results)
    n_total = len(results)
    table.add_row(
        "[bold]TOTAL[/bold]", str(n_total),
        f"[bold]{_pct(total_r, n_total)}[/bold]",
        f"[bold]{_pct(total_m, n_total)}[/bold]",
        "[bold green]MDA[/bold green]" if total_m > total_r else
        ("[bold red]RAG[/bold red]" if total_r > total_m else "[bold]TIE[/bold]"),
    )
    console.print(table)


def print_final_report(
    all_results: dict[int, list[dict]],
    checkpoints: list[int],
    rag_window: int,
) -> None:
    console.rule("[bold]Final Comparison — RAG vs MDA across Turns[/bold]")
    console.print(f"[dim]RAG context window: {rag_window:,} chars[/dim]\n")

    # Overall accuracy table: rows=clusters, cols=checkpoints
    clusters = ["A-early", "B-mid", "C-late", "cross"]
    cluster_labels = {
        "A-early": "A-early (facts @ turns  1-10)",
        "B-mid":   "B-mid   (facts @ turns 25-34)",
        "C-late":  "C-late  (facts @ turns 100-109)",
        "cross":   "Cross-cluster",
    }

    table = Table(show_header=True, header_style="bold cyan", title="Accuracy % by Cluster × Checkpoint")
    table.add_column("Cluster", width=24)
    for cp in checkpoints:
        table.add_column(f"T={cp} RAG", width=10, justify="center")
        table.add_column(f"T={cp} MDA", width=10, justify="center")

    for cluster in clusters:
        row = [cluster_labels[cluster]]
        for cp in checkpoints:
            results = all_results.get(cp, [])
            subset  = [r for r in results if r["cluster"] == cluster]
            n       = len(subset)
            rs      = sum(r["rag"]["score"] for r in subset)
            ms      = sum(r["mda"]["score"] for r in subset)
            row.append(_pct(rs, n))
            row.append(_pct(ms, n))
        table.add_row(*row)

    # TOTAL row
    row = ["[bold]TOTAL[/bold]"]
    for cp in checkpoints:
        results = all_results.get(cp, [])
        n   = len(results)
        rs  = sum(r["rag"]["score"] for r in results)
        ms  = sum(r["mda"]["score"] for r in results)
        row.append(f"[bold]{_pct(rs, n)}[/bold]")
        row.append(f"[bold]{_pct(ms, n)}[/bold]")
    table.add_row(*row)
    console.print(table)

    # Narrative analysis
    console.print("\n[bold]Key observations:[/bold]")
    for cluster in clusters:
        first_cp = checkpoints[0]
        last_cp  = checkpoints[-1]
        r_first  = [r for r in all_results.get(first_cp, []) if r["cluster"] == cluster]
        r_last   = [r for r in all_results.get(last_cp,  []) if r["cluster"] == cluster]
        if not r_first or not r_last:
            continue
        n_first = len(r_first)
        n_last  = len(r_last)
        rag_start = sum(r["rag"]["score"] for r in r_first) / (n_first * 2) * 100
        rag_end   = sum(r["rag"]["score"] for r in r_last)  / (n_last  * 2) * 100
        mda_start = sum(r["mda"]["score"] for r in r_first) / (n_first * 2) * 100
        mda_end   = sum(r["mda"]["score"] for r in r_last)  / (n_last  * 2) * 100
        rag_delta = rag_end - rag_start
        mda_delta = mda_end - mda_start
        rag_color = "red" if rag_delta < -5 else ("green" if rag_delta > 5 else "dim")
        mda_color = "green" if mda_delta > -5 else "red"
        console.print(
            f"  {cluster_labels[cluster]}: "
            f"RAG [{rag_color}]{rag_start:.0f}%→{rag_end:.0f}% ({rag_delta:+.0f})[/{rag_color}]  "
            f"MDA [{mda_color}]{mda_start:.0f}%→{mda_end:.0f}% ({mda_delta:+.0f})[/{mda_color}]"
        )


def save_results(
    all_results: dict[int, list[dict]],
    output_dir: str,
    model: str,
) -> None:
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace(":", "-").replace("/", "-")
    path = out / f"long_context_{slug}_{ts}.json"
    path.write_text(
        json.dumps(
            {str(k): v for k, v in all_results.items()},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Results saved → {path}[/dim]")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-Context Benchmark: MDA vs RAG at 10/50/200 turns"
    )
    parser.add_argument("--model",       default="qwen3:4b",
                        help="LLM for both systems and judge (default: qwen3:4b)")
    parser.add_argument("--provider",    default="ollama", choices=["ollama", "anthropic"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--api-key",     default="",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--judge",       default=None,
                        help="Judge model (defaults to --model)")
    parser.add_argument("--judge-provider", default=None, choices=["ollama", "anthropic"],
                        help="Judge provider (defaults to --provider)")
    parser.add_argument("--rag-window",  type=int, default=15_000,
                        help="Max chars in RAG context window (default: 15000 ≈ 3750 tokens)")
    parser.add_argument("--output",      default="results",
                        help="Output directory for JSON results (default: results)")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[10, 50, 200],
                        help="Turn counts at which to measure accuracy (default: 10 50 200)")
    args = parser.parse_args()

    checkpoints   = sorted(args.checkpoints)
    judge_model   = args.judge or args.model
    judge_prov    = args.judge_provider or args.provider
    api_key       = args.api_key

    console.rule("[bold]Long-Context Benchmark: MDA vs RAG[/bold]")
    console.print(f"[dim]  model: {args.model}  provider: {args.provider}[/dim]")
    console.print(f"[dim]  judge: {judge_model}  judge-provider: {judge_prov}[/dim]")
    console.print(f"[dim]  RAG context window: {args.rag_window:,} chars[/dim]")
    console.print(f"[dim]  checkpoints: {checkpoints}[/dim]\n")

    script = build_conversation_script()

    console.print("[bold yellow]Initialising MDA engine...[/bold yellow]")
    mda = build_mda_engine(args.model, args.provider, api_key)

    console.print("[bold yellow]Initialising RAG bot...[/bold yellow]")
    rag = NaiveRAGBot(
        model=args.model,
        provider=args.provider,
        api_key=api_key,
        max_context_chars=args.rag_window,
    )

    all_results:  dict[int, list[dict]] = {}
    last_turn = 0

    for cp in checkpoints:
        if cp > 200:
            console.print(f"[yellow]Checkpoint {cp} exceeds 200 turns — skipping.[/yellow]")
            continue
        # Advance conversation from last checkpoint to this one
        if cp > last_turn:
            console.print(f"\n[bold yellow]Running turns {last_turn + 1}–{cp}...[/bold yellow]")
            t0 = time.time()
            run_turns(mda, rag, script, start=last_turn, end=cp)
            console.print(f"[dim]  Done in {time.time() - t0:.1f}s[/dim]")
            last_turn = cp

        console.print(f"\n[bold yellow]Measuring accuracy at turn {cp}...[/bold yellow]")
        console.print(
            f"[dim]  RAG history: {rag.history_chars:,} chars | "
            f"visible messages: {rag.visible_turns} | "
            f"MDA entities: {len(mda.mda.registry._entities)}[/dim]"
        )
        t0 = time.time()
        results = measure_accuracy(
            mda, rag, TEST_CASES, judge_model, judge_prov, api_key, cp
        )
        console.print(f"[dim]  Accuracy measured in {time.time() - t0:.1f}s[/dim]")
        all_results[cp] = results
        print_checkpoint_summary(results, cp)

    print_final_report(all_results, checkpoints, args.rag_window)
    save_results(all_results, args.output, args.model)


if __name__ == "__main__":
    main()
