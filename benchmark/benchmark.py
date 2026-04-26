"""
MDA vs RAG (ChromaDB) Benchmark
Associative memory vs vector retrieval comparison.

Usage:
    python benchmark.py --md memory/mda.md --model qwen3:4b
    python benchmark.py --md memory/mda.md --embed nomic-embed-text --judge qwen3:4b

    # Anthropic — embed defaults to local (sentence-transformers), Ollama not required
    python benchmark.py --md .memory/mda.md --model claude-haiku-4-5-20251001 --provider anthropic

    # Anthropic + explicit local embed
    python benchmark.py --md .memory/mda.md --model claude-haiku-4-5-20251001 --provider anthropic --embed-provider local

    # Ollama LLM + local embed (when Ollama embed model is not installed)
    python benchmark.py --md .memory/mda.md --model qwen3:4b --embed-provider local
"""

import argparse
import json
import time
import re
import dotenv
dotenv.load_dotenv()  # Load variables like ANTHROPIC_API_KEY from .env
from datetime import datetime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to PYTHONPATH
import chromadb
from chromadb.utils import embedding_functions
import ollama
from rich.console import Console
from rich.table import Table

from integrations.engine import MDAEngine, AnthropicEngine




# ── LLM router ─────────────────────────────────────────────────────────────────

def _llm_complete(model: str, messages: list[dict],
                  provider: str = "ollama", api_key: str = "") -> str:
    """Unified LLM call — routes to Ollama or Anthropic."""
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package not installed — run: pip install anthropic")
        key = api_key or __import__("os").environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        system_content = ""
        user_msgs: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                user_msgs.append(m)
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_content,
            messages=user_msgs,
        )
        raw = resp.content[0].text
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw
    else:
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"].strip()

console = Console()


# ── Test seti üretimi ──────────────────────────────────────────────────────────

def generate_test_cases(md_paths: list[str], judge_model: str, questions_per_file: int = 8,
                        provider: str = "ollama", api_key: str = "") -> list[dict]:
    all_cases = []

    for md_path in md_paths:
        content = Path(md_path).read_text(encoding="utf-8")

        # Skip files that are too short — not enough content
        if len(content.strip()) < 200:
            console.print(f"[dim]  Skipping {md_path} (too short)[/dim]")
            continue

        label = Path(md_path).stem.upper()
        n_each = max(1, questions_per_file // 4)

        prompt = f"""Create a benchmark test set from the document below. Generate exactly {questions_per_file} questions.

Mix these types equally ({n_each} each):
- "fact": specific number, name, or parameter from the document
- "concept": what something is or how it works
- "reasoning": how two concepts relate to each other
- "multi-hop": chain involving three or more concepts

Rules:
- Questions must be answerable from this document only
- Each question needs 2-4 keywords expected in a correct answer
- Each question needs a concise expected_answer (1-2 sentences) taken directly from the document
- Output ONLY a JSON array, nothing else

[
  {{"type": "fact", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "concept", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "reasoning", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "multi-hop", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}}
]

Document:
{content[:6000]}"""

        def _parse_response(raw: str) -> list:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            # Greedy: take between first '[' and last ']' — outermost array
            start = raw.find("[")
            end   = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"No JSON array found: {raw[:200]}")
            candidate = raw[start:end + 1]
            parsed = json.loads(candidate)
            if not isinstance(parsed, list) or not parsed or not isinstance(parsed[0], dict):
                raise ValueError(f"Parsed result is not a list of objects: {candidate[:200]}")
            return parsed

        def _do_request(prompt_text: str) -> list:
            raw = _llm_complete(judge_model, [{"role": "user", "content": prompt_text}],
                                provider=provider, api_key=api_key)
            return _parse_response(raw)

        try:
            try:
                parsed = _do_request(prompt)
            except (json.JSONDecodeError, ValueError):
                console.print(f"[dim]  Retrying {md_path}...[/dim]")
                retry_prompt = (
                    "Output ONLY a valid JSON array starting with [ and ending with ]. "
                    "No explanation, no markdown, no extra text.\n\n" + prompt
                )
                parsed = _do_request(retry_prompt)

            for i, item in enumerate(parsed[:questions_per_file]):
                all_cases.append({
                    "id": f"{label}{i + 1}",
                    "group": label,
                    "type": item.get("type", "unknown"),
                    "question": item["question"],
                    "expected_keywords": item.get("keywords", []),
                    "expected_answer": item.get("expected_answer", ""),
                })
            console.print(f"[dim]  {min(len(parsed), questions_per_file)} questions generated from {md_path}[/dim]")
        except Exception as e:
            console.print(f"[red]  Failed to generate questions from {md_path}: {e}[/red]")

    return all_cases


# ── RAG setup (ChromaDB) ────────────────────────────────────────────────────────

def build_rag(md_paths: list[str], embed_model: str,
              embed_provider: str = "ollama") -> chromadb.Collection:
    client = chromadb.Client()

    if embed_provider == "local":
        # sentence-transformers — no Ollama needed, works with any LLM provider
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model,
        )
        embed_label = f"local:{embed_model}"
    else:
        ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=embed_model,
        )
        embed_label = f"ollama:{embed_model}"

    collection = client.get_or_create_collection("benchmark", embedding_function=ef)

    all_chunks = []
    for md_path in md_paths:
        all_chunks.extend(_chunk_md(md_path))

    if not all_chunks:
        raise ValueError("No chunks extracted from markdown files")

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(documents=all_chunks, ids=ids)
    console.print(f"[dim]  {len(all_chunks)} chunks indexed (embed: {embed_label})[/dim]")
    return collection


def rag_answer(collection: chromadb.Collection, question: str, model: str,
               provider: str = "ollama", api_key: str = "") -> tuple[str, str]:
    results = collection.query(query_texts=[question], n_results=5)
    docs = results["documents"][0] if results["documents"] else []
    context = "\n".join(docs)

    prompt = f"""You have access to the following knowledge:

{context}

Answer the question concisely based on the knowledge above.
If the knowledge does not contain relevant information, answer from your general knowledge.

Question: {question}"""

    answer = _llm_complete(model, [{"role": "user", "content": prompt}],
                           provider=provider, api_key=api_key)
    return answer, context


def _chunk_md(md_path: str) -> list[str]:
    text = Path(md_path).read_text(encoding="utf-8")
    chunks = []
    current: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                chunk = " ".join(current).strip()
                if len(chunk) >= 30:
                    chunks.append(chunk)
                current = []
        elif stripped.startswith("#"):
            if current:
                chunk = " ".join(current).strip()
                if len(chunk) >= 30:
                    chunks.append(chunk)
                current = []
            current = [stripped.lstrip("#").strip()]
        elif stripped.startswith(("```", "|", "---", ">")):
            continue
        else:
            current.append(stripped)

    if current:
        chunk = " ".join(current).strip()
        if len(chunk) >= 30:
            chunks.append(chunk)

    return chunks


# ── MDA setup ──────────────────────────────────────────────────────────────────

def build_mda(md_paths: list[str], model: str,
              provider: str = "ollama", api_key: str = "") -> MDAEngine:
    if provider == "anthropic":
        bridge = AnthropicEngine(model=model, user_id="benchmark", api_key=api_key)
    else:
        bridge = MDAEngine(model=model, user_id="benchmark")
    total = 0
    for md_path in md_paths:
        count = bridge.load_md(md_path)
        total += count
    console.print(f"[dim]  {total} facts loaded into MDA[/dim]")
    return bridge


def mda_answer(bridge: MDAEngine, question: str) -> tuple[str, str]:
    answer  = bridge.chat(question, lang="en")
    context = bridge._last_context
    return answer, context


# ── Judge ───────────────────────────────────────────────────────────────────────

def judge_answer(question: str, answer: str, expected_keywords: list[str], judge_model: str,
                 provider: str = "ollama", api_key: str = "",
                 expected_answer: str = "") -> dict:
    kw_hits  = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
    kw_score = round(kw_hits / len(expected_keywords), 2) if expected_keywords else 0.0

    if expected_answer:
        prompt = f"""Compare the given answer to the expected answer and score from 0 to 2.
2 = conveys the same information as the expected answer
1 = partially correct, missing some key details
0 = wrong, contradicts, or completely misses the point

Question: {question}
Expected answer: {expected_answer}
Given answer: {answer}

Reply with a single digit only: 0, 1, or 2. Nothing else."""
    else:
        prompt = f"""Score this AI answer from 0 to 2.
2 = correct and complete
1 = partially correct or vague
0 = wrong, hallucinated, or irrelevant

Question: {question}
Answer: {answer}

Reply with a single digit only: 0, 1, or 2. Nothing else."""

    try:
        raw = _llm_complete(judge_model, [{"role": "user", "content": prompt}],
                            provider=provider, api_key=api_key).strip()
        # strip think blocks, whitespace, punctuation
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # find first digit 0/1/2
        match = re.search(r"[012]", raw)
        if not match:
            raise ValueError(f"no digit found in: {raw[:80]!r}")
        llm_score = int(match.group(0))
        reason    = raw[:120]
    except Exception as exc:
        llm_score = -1
        reason    = f"judge failed: {exc}"

    return {
        "llm_score": llm_score,
        "kw_score":  kw_score,
        "kw_hits":   kw_hits,
        "kw_total":  len(expected_keywords),
        "reason":    reason,
    }


# ── Runner ──────────────────────────────────────────────────────────────────────

def run_benchmark(md_paths: list[str], model: str, embed_model: str, judge_model: str,
                  output_dir: str, questions_per_file: int,
                  provider: str = "ollama", api_key: str = "",
                  embed_provider: str = "ollama") -> None:
    console.rule("[bold]MDA vs RAG Benchmark[/bold]")
    console.print(f"[dim]  provider: {provider}  model: {model}  judge: {judge_model}[/dim]\n")

    console.print("\n[bold yellow]Generating test cases...[/bold yellow]")
    test_cases = generate_test_cases(md_paths, judge_model, questions_per_file,
                                     provider=provider, api_key=api_key)
    if not test_cases:
        console.print("[red]No test cases generated. Exiting.[/red]")
        return
    console.print(f"[bold]  Total: {len(test_cases)} questions[/bold]\n")

    console.print("[bold yellow]Building RAG...[/bold yellow]")
    rag = None
    try:
        rag = build_rag(md_paths, embed_model, embed_provider=embed_provider)
    except Exception as e:
        console.print(f"[red]  RAG build failed: {e}[/red]")
        console.print("[yellow]  Continuing without RAG — RAG scores will be 0.[/yellow]")

    console.print("[bold yellow]Building MDA...[/bold yellow]")
    mda = build_mda(md_paths, model, provider=provider, api_key=api_key)

    results = []
    console.print(f"\n[bold]Running {len(test_cases)} test cases...[/bold]\n")

    for tc in test_cases:
        console.print(f"[dim][{tc['id']}][/dim] {tc['question']}")

        t0 = time.time()
        if rag is not None:
            rag_ans, rag_ctx = rag_answer(rag, tc["question"], model,
                                          provider=provider, api_key=api_key)
        else:
            rag_ans, rag_ctx = "[RAG unavailable]", ""
        rag_time = round(time.time() - t0, 2)

        t0 = time.time()
        mda_ans, mda_ctx = mda_answer(mda, tc["question"])
        mda_time = round(time.time() - t0, 2)

        rag_j = judge_answer(tc["question"], rag_ans, tc["expected_keywords"], judge_model,
                             provider=provider, api_key=api_key,
                             expected_answer=tc.get("expected_answer", ""))
        mda_j = judge_answer(tc["question"], mda_ans, tc["expected_keywords"], judge_model,
                             provider=provider, api_key=api_key,
                             expected_answer=tc.get("expected_answer", ""))

        result = {
            "id": tc["id"],
            "group": tc["group"],
            "type": tc.get("type", "unknown"),
            "question": tc["question"],
            "rag": {"answer": rag_ans, "context_len": len(rag_ctx), "time_s": rag_time, **rag_j},
            "mda": {"answer": mda_ans, "context_len": len(mda_ctx), "time_s": mda_time, **mda_j},
        }
        results.append(result)

        rs = rag_j["llm_score"]
        ms = mda_j["llm_score"]
        winner = "TIE" if rs == ms else ("MDA ✓" if ms > rs else "RAG ✓")
        console.print(f"  RAG={rs}/2  [{rag_j['reason'][:60]}]")
        console.print(f"  MDA={ms}/2  [{mda_j['reason'][:60]}]")
        console.print(f"  RAG ans: {rag_ans[:80]}")
        console.print(f"  MDA ans: {mda_ans[:80]}")
        console.print(f"  MDA ctx_len={len(mda_ctx)}  → {winner}\n")

    _print_report(results)
    _save_results(results, output_dir, model)


def _print_report(results: list[dict]) -> None:
    console.rule("[bold]Results[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", width=8)
    table.add_column("Type", width=10)
    table.add_column("Question", width=38)
    table.add_column("RAG", width=5, justify="center")
    table.add_column("MDA", width=5, justify="center")
    table.add_column("Winner", width=8, justify="center")

    group_stats: dict[str, dict] = {}
    type_stats:  dict[str, dict] = {}

    for r in results:
        rs = r["rag"]["llm_score"]
        ms = r["mda"]["llm_score"]
        if rs == ms:
            winner = "[dim]TIE[/dim]"
        elif ms > rs:
            winner = "[green]MDA[/green]"
        else:
            winner = "[red]RAG[/red]"

        g = r["group"]
        if g not in group_stats:
            group_stats[g] = {"rag": 0, "mda": 0, "n": 0}
        group_stats[g]["rag"] += max(rs, 0)
        group_stats[g]["mda"] += max(ms, 0)
        group_stats[g]["n"] += 1

        t = r.get("type", "unknown")
        if t not in type_stats:
            type_stats[t] = {"rag": 0, "mda": 0, "n": 0}
        type_stats[t]["rag"] += max(rs, 0)
        type_stats[t]["mda"] += max(ms, 0)
        type_stats[t]["n"] += 1

        table.add_row(r["id"], t, r["question"][:36], str(rs), str(ms), winner)

    console.print(table)

    # Category breakdown
    console.print("\n[bold]By Question Type:[/bold]")
    type_order = ["fact", "concept", "reasoning", "multi-hop", "unknown"]
    for t in type_order:
        if t not in type_stats:
            continue
        s = type_stats[t]
        max_score = s["n"] * 2
        rag_s = s["rag"]
        mda_s = s["mda"]
        if mda_s > rag_s:
            tag = "[green]MDA ↑[/green]"
        elif rag_s > mda_s:
            tag = "[red]RAG ↑[/red]"
        else:
            tag = "[dim]TIE[/dim]"
        console.print(f"  {t:<12} RAG {rag_s}/{max_score}  MDA {mda_s}/{max_score}  {tag}")

    console.print("\n[bold]Group Summary:[/bold]")
    total_rag = total_mda = total_n = 0
    for g, s in sorted(group_stats.items()):
        max_score = s["n"] * 2
        console.print(f"  {g}: RAG {s['rag']}/{max_score}  MDA {s['mda']}/{max_score}")
        total_rag += s["rag"]
        total_mda += s["mda"]
        total_n += s["n"]

    total_max = total_n * 2
    console.print(f"\n  [bold]TOTAL: RAG {total_rag}/{total_max}  MDA {total_mda}/{total_max}[/bold]")

    if total_mda > total_rag:
        console.print("  [bold green]→ MDA wins[/bold green]")
    elif total_rag > total_mda:
        console.print("  [bold red]→ RAG wins[/bold red]")
    else:
        console.print("  [bold yellow]→ TIE[/bold yellow]")


def _save_results(results: list[dict], output_dir: str, model: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"benchmark_{model.replace(':', '-')}_{ts}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Results saved → {path}[/dim]")


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDA vs RAG Benchmark")
    parser.add_argument("--md", nargs="+", help="Knowledge base markdown files (space separated)")
    parser.add_argument("--memory", default=None, help="Auto-discover all .md files in this directory (e.g. ../.memory)")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model for answers and MDA")
    parser.add_argument("--embed", default=None, help="Embedding model (default: nomic-embed-text for ollama, all-MiniLM-L6-v2 for local)")
    parser.add_argument("--embed-provider", default=None, choices=["ollama", "local"],
                        help="Embedding provider: ollama (default) or local (sentence-transformers, no Ollama needed)")
    parser.add_argument("--judge", default=None, help="Judge model (default: same as --model)")
    parser.add_argument("--questions", type=int, default=8, help="Questions per md file (default: 8)")
    parser.add_argument("--output", default="results", help="Output directory for JSON results")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "anthropic"],
                        help="LLM provider (default: ollama)")
    parser.add_argument("--api-key", default="", help="API key (or set ANTHROPIC_API_KEY env var)")
    args = parser.parse_args()

    # --memory ile .memory/ altındaki tüm .md dosyaları otomatik bulunur
    if args.memory:
        memory_dir = Path(args.memory)
        discovered = sorted(memory_dir.rglob("*.md"))
        if not discovered:
            print(f"[red]No .md files found in {memory_dir}[/red]")
            raise SystemExit(1)
        md_paths = [str(p) for p in discovered]
        console.print(f"[dim]Auto-discovered {len(md_paths)} md files:[/dim]")
        for p in md_paths:
            console.print(f"[dim]  {p}[/dim]")
    elif args.md:
        md_paths = args.md
    else:
        parser.error("Provide either --md <files> or --memory <directory>")

    judge_model = args.judge or args.model

    # Auto-default embed provider: anthropic → local (no Ollama needed), ollama → ollama
    embed_provider = args.embed_provider or ("local" if args.provider == "anthropic" else "ollama")
    embed_model    = args.embed or ("all-MiniLM-L6-v2" if embed_provider == "local" else "nomic-embed-text")

    run_benchmark(md_paths, args.model, embed_model, judge_model, args.output, args.questions,
                  provider=args.provider, api_key=args.api_key, embed_provider=embed_provider)