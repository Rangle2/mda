"""
MDA Ultimate Benchmark
======================
80 hand-crafted questions across 8 categories and 3 domains.
Compares MDA against a production-grade RAG (bge-large + overlap chunking).

Categories:
  ATOMIC_RECALL       — specific fact retrieval
  MULTI_HOP           — 3-4 entity chain reasoning
  CROSS_DOCUMENT      — facts spanning multiple md files
  REASONING           — rule application + quantitative reasoning
  INCREMENTAL_LEARNING— online update (RAG cannot do this)
  NOISE_RESISTANCE    — misinformation / override resistance
  MEMORY_COMPRESSION  — answer quality per token used
  BOUNDARY            — hallucination resistance

Usage:
    # Ollama (fully free)
    python ultimate_benchmark.py --model qwen3:4b

    # Anthropic
    python ultimate_benchmark.py --model claude-haiku-4-5-20251001 --provider anthropic
"""

import argparse
import json
import os
import re
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import dotenv
dotenv.load_dotenv()

MDA_LIB = Path(__file__).parent.parent.parent  # benchmark/full_benchmark -> benchmark -> mda-lib root
sys.path.insert(0, str(MDA_LIB))

from mda.integrations.engine import MDAEngine, AnthropicEngine

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
except ImportError:
    class _C:
        def print(self, *a, **k): print(*a)
        def rule(self, *a, **k): print("─" * 60)
    console = _C()

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    console.print("[yellow]chromadb not found — RAG will be skipped[/yellow]")


# ── Constants ──────────────────────────────────────────────────────────────────

DOMAINS = {
    "zephyria": [
        str(Path(__file__).parent / "zephyria.md"),
    ],
    "veloria": [
        str(Path(__file__).parent / "veloria_geography.md"),
        str(Path(__file__).parent / "veloria_economy.md"),
        str(Path(__file__).parent / "veloria_science.md"),
    ],
    "requests": [
        str(Path(__file__).parent / "requests_lib.md"),
    ],
}

RAG_CANNOT = {"INCREMENTAL_LEARNING"}  # RAG gets 0 on these — by design
RAG_STRUGGLES = {"CROSS_DOCUMENT", "NOISE_RESISTANCE", "MEMORY_COMPRESSION"}


# ── LLM router ─────────────────────────────────────────────────────────────────

def _llm(model: str, messages: list[dict], provider: str, api_key: str = "") -> str:
    if provider == "anthropic":
        import anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        resp = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=512, system=system, messages=user_msgs,
        )
        raw = resp.content[0].text
    else:
        import ollama as _ollama
        raw = _ollama.chat(model=model, messages=messages)["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── RAG ────────────────────────────────────────────────────────────────────────

def _chunk_md(path: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Overlapping chunk strategy for stronger RAG."""
    text = Path(path).read_text(encoding="utf-8")
    # Strip markdown symbols
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) >= 50:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_rag(domain: str) -> object | None:
    if not HAS_CHROMA:
        return None
    paths = DOMAINS[domain]
    client = chromadb.Client()
    col_name = f"ult_{domain}_{hashlib.md5(domain.encode()).hexdigest()[:6]}"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    # Try bge-large first, fall back to MiniLM
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5"
        )
        ef_name = "bge-large"
    except Exception:
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            ef_name = "MiniLM"
        except Exception:
            return None

    col = client.get_or_create_collection(col_name, embedding_function=ef)
    all_chunks, ids = [], []
    for path in paths:
        chunks = _chunk_md(path)
        for j, c in enumerate(chunks):
            all_chunks.append(c)
            ids.append(f"{Path(path).stem}_{j}")
    col.add(documents=all_chunks, ids=ids)
    col._ef_name = ef_name
    col._chunk_count = len(all_chunks)
    return col


def rag_answer(col, question: str, model: str, provider: str, api_key: str,
               group: str) -> tuple[str, int]:
    if col is None or group in RAG_CANNOT:
        return "[RAG_SKIP — category not applicable]", 0

    docs = col.query(query_texts=[question], n_results=min(6, col._chunk_count))["documents"][0]
    ctx = "\n\n".join(docs)
    prompt = (
        f"You have access to the following knowledge:\n\n{ctx}\n\n"
        f"Answer the question based ONLY on the knowledge above. "
        f"If the answer is not there, say 'I don't know'.\n\n"
        f"Question: {question}"
    )
    ans = _llm(model, [{"role": "user", "content": prompt}], provider, api_key)
    return ans, len(ctx)


# ── MDA ────────────────────────────────────────────────────────────────────────

def build_mda(domain: str, model: str, provider: str, api_key: str):
    if provider == "anthropic":
        engine = AnthropicEngine(model=model, user_id=f"ult_{domain}", api_key=api_key)
    else:
        engine = MDAEngine(model=model, user_id=f"ult_{domain}")
    for path in DOMAINS[domain]:
        engine.load_md(path)
    return engine


def mda_answer(engine, question: str) -> tuple[str, int]:
    ans = engine.chat(question, lang="en")
    ctx_len = len(getattr(engine, "_last_context", ""))
    return ans, ctx_len


# ── Judge ──────────────────────────────────────────────────────────────────────

def judge(question: str, answer: str, expected_keywords: list[str],
          expected_answer: str, model: str, provider: str, api_key: str,
          group: str, judge_model: str = "", judge_provider: str = "",
          judge_api_key: str = "") -> dict:
    # RAG_CANNOT categories — RAG always scores 0
    if group in RAG_CANNOT and answer.startswith("[RAG_SKIP"):
        return {"llm_score": 0, "kw_score": 0.0, "reason": "RAG cannot do incremental learning"}

    kw_hits  = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    kw_score = round(kw_hits / len(expected_keywords), 2) if expected_keywords else None

    boundary_hint = ""
    if group == "BOUNDARY":
        boundary_hint = (
            "SPECIAL RULE: If the expected answer says the system should say it does NOT know, "
            "score 2 if the answer says it doesn't know, score 0 if it fabricates an answer. "
        )

    noise_hint = ""
    if group == "NOISE_RESISTANCE":
        noise_hint = (
            "SPECIAL RULE: Score 2 if the answer correctly rejects the false claim and states "
            "the accurate fact from the knowledge base. Score 0 if it accepts the false claim. "
        )

    prompt = (
        f"Score this answer: 0, 1, or 2.\n"
        f"2 = fully correct\n1 = partially correct\n0 = wrong, fabricated, or misses the point\n"
        f"{boundary_hint}{noise_hint}"
        f"If the given answer contains the correct numeric value or key term anywhere in the text, score 2 even if the phrasing differs from the expected answer.\n"
        f"Question: {question}\n"
        f"Expected: {expected_answer}\n"
        f"Given: {answer}\n\n"
        f"Reply with a single digit only: 0, 1, or 2."
    )
    try:
        jm  = judge_model or model
        jp  = judge_provider or provider
        jak = judge_api_key or api_key
        raw   = _llm(jm, [{"role": "user", "content": prompt}], jp, jak)
        m     = re.search(r"[012]", raw)
        score = int(m.group(0)) if m else -1
    except Exception as e:
        score, raw = -1, str(e)
    return {"llm_score": score, "kw_score": kw_score, "kw_hits": kw_hits,
            "kw_total": len(expected_keywords), "reason": raw[:80]}


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(test_path: str, model: str, provider: str, api_key: str,
        output_dir: str, domains_filter: list[str] | None = None,
        judge_model: str = "", judge_provider: str = "",
        judge_api_key: str = "") -> None:

    console.rule("[bold]MDA Ultimate Benchmark[/bold]")
    jm_info = f"  judge={judge_model or model} ({judge_provider or provider})" if judge_model else ""
    console.print(f"[dim]  provider={provider}  model={model}{jm_info}[/dim]\n")

    test_cases = json.loads(Path(test_path).read_text(encoding="utf-8"))
    if domains_filter:
        test_cases = [t for t in test_cases if t["domain"] in domains_filter]
    console.print(f"  [bold]{len(test_cases)} test cases[/bold] across "
                  f"{len(set(t['domain'] for t in test_cases))} domains\n")

    # Build systems per domain
    console.print("[bold yellow]Building RAG indexes...[/bold yellow]")
    rag_systems: dict[str, object] = {}
    for domain in set(t["domain"] for t in test_cases):
        rag_systems[domain] = build_rag(domain)
        ef = getattr(rag_systems[domain], "_ef_name", "N/A")
        chunks = getattr(rag_systems[domain], "_chunk_count", 0)
        console.print(f"  {domain}: {chunks} chunks, embed={ef}")

    console.print("\n[bold yellow]Building MDA instances...[/bold yellow]")
    mda_systems: dict[str, object] = {}
    for domain in set(t["domain"] for t in test_cases):
        mda_systems[domain] = build_mda(domain, model, provider, api_key)
        console.print(f"  {domain}: loaded")

    results = []
    group_stats: dict[str, dict] = {}

    console.print(f"\n[bold]Running {len(test_cases)} questions...[/bold]\n")

    for tc in test_cases:
        qid      = tc["id"]
        group    = tc["group"]
        domain   = tc["domain"]
        question = tc["question"]
        exp_kw   = tc.get("expected_keywords", [])
        exp_ans  = tc.get("expected_answer", "")
        measures = tc.get("measures", "")

        console.print(f"[dim][{qid}][/dim] [bold]{group}[/bold] ({domain})")
        console.print(f"  {question[:100]}")

        t0 = time.time()
        rag_ans, rag_ctx_len = rag_answer(
            rag_systems[domain], question, model, provider, api_key, group
        )
        rag_t = round(time.time() - t0, 2)

        t0 = time.time()
        mda_ans, mda_ctx_len = mda_answer(mda_systems[domain], question)
        mda_t = round(time.time() - t0, 2)

        rag_j = judge(question, rag_ans, exp_kw, exp_ans, model, provider, api_key, group,
                      judge_model, judge_provider, judge_api_key)
        mda_j = judge(question, mda_ans, exp_kw, exp_ans, model, provider, api_key, group,
                      judge_model, judge_provider, judge_api_key)

        rs, ms = rag_j["llm_score"], mda_j["llm_score"]
        winner = "TIE" if rs == ms else ("MDA ✓" if ms > rs else "RAG ✓")

        ctx_efficiency = ""
        if rs == ms and ms > 0 and rag_ctx_len > 0 and mda_ctx_len > 0:
            ratio = rag_ctx_len / mda_ctx_len
            ctx_efficiency = f"  [dim]ctx: RAG={rag_ctx_len} MDA={mda_ctx_len} ratio={ratio:.1f}x[/dim]"

        console.print(f"  RAG={rs}/2  MDA={ms}/2  → {winner}{ctx_efficiency}")
        console.print(f"  measures: [italic]{measures}[/italic]\n")

        if group not in group_stats:
            group_stats[group] = {
                "rag": 0, "mda": 0, "n": 0,
                "mda_wins": 0, "rag_wins": 0, "ties": 0,
                "rag_ctx": 0, "mda_ctx": 0, "ctx_n": 0,
            }
        s = group_stats[group]
        s["rag"] += max(rs, 0)
        s["mda"] += max(ms, 0)
        s["n"]   += 1
        if ms > rs: s["mda_wins"] += 1
        elif rs > ms: s["rag_wins"] += 1
        else: s["ties"] += 1
        if rag_ctx_len > 0 and mda_ctx_len > 0:
            s["rag_ctx"] += rag_ctx_len
            s["mda_ctx"] += mda_ctx_len
            s["ctx_n"]   += 1

        results.append({
            "id": qid, "group": group, "domain": domain,
            "question": question, "measures": measures,
            "rag": {"answer": rag_ans, "context_len": rag_ctx_len, "time_s": rag_t, **rag_j},
            "mda": {"answer": mda_ans, "context_len": mda_ctx_len, "time_s": mda_t, **mda_j},
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    console.rule("[bold]Summary[/bold]")
    total_n   = sum(v["n"]   for v in group_stats.values())
    total_rag = sum(v["rag"] for v in group_stats.values())
    total_mda = sum(v["mda"] for v in group_stats.values())
    max_score = total_n * 2
    total_rag_ctx = sum(v["rag_ctx"] for v in group_stats.values())
    total_mda_ctx = sum(v["mda_ctx"] for v in group_stats.values())
    ctx_n     = sum(v["ctx_n"] for v in group_stats.values())

    console.print(f"\n[bold]OVERALL[/bold]")
    console.print(f"  RAG: {total_rag}/{max_score} ({total_rag/max_score*100:.1f}%)")
    console.print(f"  MDA: {total_mda}/{max_score} ({total_mda/max_score*100:.1f}%)")
    if ctx_n > 0:
        console.print(f"\n[bold]TOKEN EFFICIENCY (same-score questions)[/bold]")
        console.print(f"  RAG avg context: {total_rag_ctx//ctx_n:,} chars")
        console.print(f"  MDA avg context: {total_mda_ctx//ctx_n:,} chars")
        console.print(f"  MDA uses {total_rag_ctx/total_mda_ctx:.1f}x less context on average")

    console.print(f"\n[bold]BY CATEGORY[/bold]")
    RAG_LABEL = {
        "ATOMIC_RECALL":       "Atomic Recall      (both should handle)",
        "MULTI_HOP":           "Multi-Hop          (MDA advantage)",
        "CROSS_DOCUMENT":      "Cross-Document     (MDA advantage)",
        "REASONING":           "Reasoning          (MDA advantage)",
        "INCREMENTAL_LEARNING":"Incremental Learn  (RAG cannot do)",
        "NOISE_RESISTANCE":    "Noise Resistance   (MDA advantage)",
        "MEMORY_COMPRESSION":  "Memory Compression (token efficiency)",
        "BOUNDARY":            "Boundary           (hallucination resist)",
    }
    for g, label in RAG_LABEL.items():
        if g not in group_stats:
            continue
        v  = group_stats[g]
        mx = v["n"] * 2
        rp = v["rag"] / mx * 100
        mp = v["mda"] / mx * 100
        d  = mp - rp
        sign = "+" if d >= 0 else ""
        ctx_str = ""
        if v["ctx_n"] > 0:
            r_avg = v["rag_ctx"] // v["ctx_n"]
            m_avg = v["mda_ctx"] // v["ctx_n"]
            ctx_str = f"  ctx RAG={r_avg} MDA={m_avg}"
        console.print(
            f"  {label}\n"
            f"    RAG {v['rag']}/{mx} ({rp:.0f}%)  "
            f"MDA {v['mda']}/{mx} ({mp:.0f}%)  "
            f"delta={sign}{d:.0f}%  "
            f"MDA↑={v['mda_wins']} RAG↑={v['rag_wins']}"
            f"{ctx_str}"
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace(":", "-").replace("/", "-")
    out  = Path(output_dir) / f"ultimate_{slug}_{ts}.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Results → {out}[/dim]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _args():
    p = argparse.ArgumentParser(description="MDA Ultimate Benchmark")
    p.add_argument("--tests",    default=str(Path(__file__).parent / "ultimate_benchmark.json"))
    p.add_argument("--model",    default="qwen3:4b")
    p.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"])
    p.add_argument("--api-key",  default="", dest="api_key")
    p.add_argument("--output",   default=str(MDA_LIB / "results"))
    p.add_argument("--domains",  nargs="+", choices=["zephyria", "veloria", "requests"],
                   help="Run only specific domains")
    p.add_argument("--judge-model",    default="", dest="judge_model",
                   help="Separate model for judging (default: same as --model)")
    p.add_argument("--judge-provider", default="", dest="judge_provider",
                   choices=["", "ollama", "anthropic"],
                   help="Provider for judge model")
    p.add_argument("--judge-api-key",  default="", dest="judge_api_key")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    run(
        test_path      = a.tests,
        model          = a.model,
        provider       = a.provider,
        api_key        = a.api_key,
        output_dir     = a.output,
        domains_filter = a.domains,
        judge_model    = a.judge_model,
        judge_provider = a.judge_provider,
        judge_api_key  = a.judge_api_key,
    )