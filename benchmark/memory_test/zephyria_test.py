"""
Zephyria Memory Benchmark Runner
Measures MDA memory system behaviour across 5 dimensions:
  ATOMIC_RECALL   — specific number/name recall
  ASSOCIATION     — link between two entities
  MULTI_HOP       — chain of 3+ entities
  REASONING       — rule application, comparison
  CONFLICT_UPDATE — update / counterfactual
  BOUNDARY        — hallucination resistance (things the LLM should not know)

Usage:
    python zephyria_test.py --provider anthropic --model claude-haiku-4-5-20251001
    python zephyria_test.py --provider ollama --model qwen3:4b
"""

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

# Add mda-lib root to path
SCRIPT_DIR = Path(__file__).parent
MDA_LIB    = SCRIPT_DIR.parent.parent  # test/memory_test -> test -> mda-lib root
sys.path.insert(0, str(MDA_LIB))

from integrations.engine import MDAEngine, AnthropicEngine

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
        def rule(self, *a, **k): print("─" * 60)
    console = _FallbackConsole()
    Table = None


# ── LLM router ─────────────────────────────────────────────────────────────────

def _llm(model: str, messages: list[dict], provider: str, api_key: str = "") -> str:
    if provider == "anthropic":
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=512,
            system=system, messages=user_msgs,
        )
        raw = resp.content[0].text
    else:
        import ollama as _ollama
        raw = _ollama.chat(model=model, messages=messages)["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── RAG ────────────────────────────────────────────────────────────────────────

def _chunk_md(path: str) -> list[str]:
    chunks, cur = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            if cur:
                c = " ".join(cur)
                if len(c) >= 30: chunks.append(c)
                cur = []
        elif s.startswith("#"):
            if cur:
                c = " ".join(cur)
                if len(c) >= 30: chunks.append(c)
            cur = [s.lstrip("#").strip()]
        elif s.startswith(("```", "|", "---", ">")):
            continue
        else:
            cur.append(s)
    if cur:
        c = " ".join(cur)
        if len(c) >= 30: chunks.append(c)
    return chunks


def build_rag(md_path: str):
    if not HAS_CHROMA:
        return None
    client = chromadb.Client()
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        return None
    col = client.get_or_create_collection("zephyria_bench", embedding_function=ef)
    chunks = _chunk_md(md_path)
    col.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])
    console.print(f"[dim]  RAG: {len(chunks)} chunks indexed[/dim]")
    return col


def rag_answer(col, question: str, model: str, provider: str, api_key: str) -> tuple[str, str]:
    if col is None:
        return "[RAG unavailable]", ""
    docs = col.query(query_texts=[question], n_results=5)["documents"][0]
    ctx  = "\n".join(docs)
    prompt = (
        f"You have access to the following knowledge:\n\n{ctx}\n\n"
        f"Answer the question based ONLY on the knowledge above. "
        f"If the answer is not in the knowledge, say 'I don't know'.\n\n"
        f"Question: {question}"
    )
    ans = _llm(model, [{"role": "user", "content": prompt}], provider, api_key)
    return ans, ctx


# ── MDA ────────────────────────────────────────────────────────────────────────

def build_mda(md_path: str, model: str, provider: str, api_key: str) -> MDAEngine:
    if provider == "anthropic":
        bridge = AnthropicEngine(model=model, user_id="zephyria_bench", api_key=api_key)
    else:
        bridge = MDAEngine(model=model, user_id="zephyria_bench")
    n = bridge.load_md(md_path)
    console.print(f"[dim]  MDA: {n} facts loaded[/dim]")
    return bridge


def mda_answer(bridge: MDAEngine, question: str) -> tuple[str, str]:
    ans = bridge.chat(question, lang="en")
    return ans, bridge._last_context


# ── Judge ──────────────────────────────────────────────────────────────────────

def judge(question: str, answer: str, expected_keywords: list[str],
          expected_answer: str, model: str, provider: str, api_key: str) -> dict:
    kw_hits  = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    kw_score = round(kw_hits / len(expected_keywords), 2) if expected_keywords else None

    prompt = (
        f"Compare the given answer to the expected answer and score 0, 1, or 2.\n"
        f"2 = conveys the same key information\n"
        f"1 = partially correct, missing important details\n"
        f"0 = wrong, contradicts, hallucinates, or completely misses the point\n\n"
        f"Special case: if the question is a BOUNDARY test (the expected answer says "
        f"the system should say it does not know), score 2 if the answer says it doesn't "
        f"know, score 0 if it makes up an answer.\n\n"
        f"If the given answer contains the correct numeric value or key term anywhere in the text, score 2 even if the phrasing differs from the expected answer.\n"
        f"Question: {question}\n"
        f"Expected: {expected_answer}\n"
        f"Given: {answer}\n\n"
        f"Reply with a single digit only: 0, 1, or 2."
    )
    try:
        raw   = _llm(model, [{"role": "user", "content": prompt}], provider, api_key)
        m     = re.search(r"[012]", raw)
        score = int(m.group(0)) if m else -1
    except Exception as e:
        score = -1
        raw   = str(e)
    return {"llm_score": score, "kw_score": kw_score,
            "kw_hits": kw_hits, "kw_total": len(expected_keywords), "reason": raw[:80]}


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(md_path: str, test_path: str, model: str, provider: str, api_key: str,
        output_dir: str) -> None:
    console.rule("[bold]Zephyria Memory Benchmark[/bold]")
    console.print(f"[dim]  provider={provider}  model={model}[/dim]\n")

    test_cases = json.loads(Path(test_path).read_text(encoding="utf-8"))
    console.print(f"[bold]  {len(test_cases)} test cases loaded[/bold]\n")

    console.print("[bold yellow]Building RAG...[/bold yellow]")
    rag = build_rag(md_path)
    if rag is None:
        console.print("[yellow]  RAG skipped (chromadb/sentence-transformers not available)[/yellow]")

    console.print("[bold yellow]Building MDA...[/bold yellow]")
    mda = build_mda(md_path, model, provider, api_key)

    results = []
    console.print(f"\n[bold]Running {len(test_cases)} questions...[/bold]\n")

    group_stats: dict[str, dict] = {}

    for tc in test_cases:
        qid      = tc["id"]
        group    = tc["group"]
        question = tc["question"]
        exp_kw   = tc.get("expected_keywords", [])
        exp_ans  = tc.get("expected_answer", "")
        measures = tc.get("measures", "")

        console.print(f"[dim][{qid}][/dim] {question}")

        t0 = time.time()
        rag_ans, rag_ctx = rag_answer(rag, question, model, provider, api_key)
        rag_t = round(time.time() - t0, 2)

        t0 = time.time()
        mda_ans, mda_ctx = mda_answer(mda, question)
        mda_t = round(time.time() - t0, 2)

        rag_j = judge(question, rag_ans, exp_kw, exp_ans, model, provider, api_key)
        mda_j = judge(question, mda_ans, exp_kw, exp_ans, model, provider, api_key)

        rs, ms = rag_j["llm_score"], mda_j["llm_score"]
        winner = "TIE" if rs == ms else ("MDA ✓" if ms > rs else "RAG ✓")

        console.print(f"  measures: [italic]{measures}[/italic]")
        console.print(f"  RAG={rs}/2  {rag_ans[:90]}")
        console.print(f"  MDA={ms}/2  {mda_ans[:90]}  ctx={len(mda_ctx)}")
        console.print(f"  → {winner}\n")

        if group not in group_stats:
            group_stats[group] = {"rag": 0, "mda": 0, "n": 0, "mda_wins": 0, "rag_wins": 0}
        group_stats[group]["rag"] += max(rs, 0)
        group_stats[group]["mda"] += max(ms, 0)
        group_stats[group]["n"]   += 1
        if ms > rs: group_stats[group]["mda_wins"] += 1
        if rs > ms: group_stats[group]["rag_wins"] += 1

        results.append({
            "id": qid, "group": group, "type": tc.get("type", ""),
            "question": question, "measures": measures,
            "rag": {"answer": rag_ans, "context_len": len(rag_ctx), "time_s": rag_t, **rag_j},
            "mda": {"answer": mda_ans, "context_len": len(mda_ctx), "time_s": mda_t, **mda_j},
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    console.rule("[bold]Summary[/bold]")
    total_n   = sum(v["n"]   for v in group_stats.values())
    total_rag = sum(v["rag"] for v in group_stats.values())
    total_mda = sum(v["mda"] for v in group_stats.values())
    max_score = total_n * 2

    console.print(f"\nOverall — RAG: {total_rag}/{max_score} ({total_rag/max_score*100:.1f}%)  "
                  f"MDA: {total_mda}/{max_score} ({total_mda/max_score*100:.1f}%)\n")

    GROUP_LABELS = {
        "ATOMIC_RECALL":   "Atomic Recall   (specific numbers/names)",
        "ASSOCIATION":     "Association     (two-entity links)",
        "MULTI_HOP":       "Multi-Hop       (3+ entity chains)",
        "REASONING":       "Reasoning       (rules, comparison)",
        "CONFLICT_UPDATE": "Conflict/Update (corrections, counterfactuals)",
        "BOUNDARY":        "Boundary        (hallucination resistance)",
    }

    for g, label in GROUP_LABELS.items():
        if g not in group_stats:
            continue
        v  = group_stats[g]
        mx = v["n"] * 2
        rag_pct = v["rag"] / mx * 100
        mda_pct = v["mda"] / mx * 100
        delta   = mda_pct - rag_pct
        sign    = "+" if delta >= 0 else ""
        console.print(f"  {label}")
        console.print(f"    RAG {v['rag']}/{mx} ({rag_pct:.0f}%)  "
                      f"MDA {v['mda']}/{mx} ({mda_pct:.0f}%)  "
                      f"delta={sign}{delta:.0f}%  "
                      f"MDA wins={v['mda_wins']}  RAG wins={v['rag_wins']}")

    # ── Save ───────────────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace(":", "-").replace("/", "-")
    out  = Path(output_dir) / f"zephyria_{slug}_{ts}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Results saved → {out}[/dim]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="Zephyria Memory Benchmark")
    p.add_argument("--md",       default=str(SCRIPT_DIR / "zephyria.md"),
                   help="Path to Zephyria knowledge file")
    p.add_argument("--tests",    default=str(SCRIPT_DIR / "zephyria_benchmark.json"),
                   help="Path to hand-crafted test cases JSON")
    p.add_argument("--model",    default="claude-haiku-4-5-20251001",
                   help="LLM model name")
    p.add_argument("--provider", default="anthropic", choices=["ollama", "anthropic"])
    p.add_argument("--api-key",  default="", dest="api_key")
    p.add_argument("--output",   default=str(MDA_LIB / "results"),
                   help="Output directory for results JSON")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    run(
        md_path    = a.md,
        test_path  = a.tests,
        model      = a.model,
        provider   = a.provider,
        api_key    = a.api_key,
        output_dir = a.output,
    )