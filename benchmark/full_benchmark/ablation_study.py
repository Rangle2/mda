"""
MDA Ablation Study
Runs the benchmark under 4 configurations to isolate each component's contribution:

  full      — full MDA (baseline)
  no_hdr    — HDR encoding replaced with random Gaussian vectors (encoding off)
  no_graph  — synapse graph traversal disabled (only origin entity facts used)
  no_oja    — Oja/W-matrix learning disabled (W always None, query cosine only)

Usage:
    python benchmark/ablation_study.py --md .memory/mda.md --model qwen3:4b
    python benchmark/ablation_study.py --md .memory/mda.md \\
        --model claude-haiku-4-5-20251001 --provider anthropic
    python benchmark/ablation_study.py --md .memory/mda.md --model qwen3:4b \\
        --configs full no_oja          # run only two configs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

# ── Path fix: must happen before ANY local mda import ──────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import dotenv
import numpy as np

dotenv.load_dotenv()

from rich.console import Console
from rich.table import Table

from mda.integrations.engine import MDAEngine, AnthropicEngine
from mda.core.bind import normalize, cosine, DIM
from mda.core.encoder import HolisticEncoder
from mda.inference.associative import AssociativeChain, ChainResult
from mda.inference.broca import BrocaModule

console = Console()

AblationMode = Literal["full", "no_hdr", "no_graph", "no_oja"]

ABLATION_LABELS = {
    "full":     "Full MDA",
    "no_hdr":   "−HDR (random enc.)",
    "no_graph": "−Graph (no traversal)",
    "no_oja":   "−Oja (static W)",
}


# ── Ablation patches ──────────────────────────────────────────────────────────

def _patch_no_hdr(engine: MDAEngine) -> None:
    """Replace encode() with random Gaussian vectors — deterministic per text."""
    original_encode = engine.mda.encoder.encode

    def _random_encode(text: str) -> np.ndarray:
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.default_rng(seed)
        return normalize(rng.normal(0, 1, DIM))

    engine.mda.encoder.encode = _random_encode
    # Also patch the broca encoder so scoring uses the same random vecs
    engine.mda.broca.encoder.encode = _random_encode


def _patch_no_graph(engine: MDAEngine) -> None:
    """Disable synapse traversal — chain returns only origin entity."""

    original_expand = engine.mda._chain.expand

    def _origin_only_expand(origin_entity, context_vec=None, query_vec=None):
        from mda.inference.associative import ChainNode, ChainResult
        node = ChainNode(
            entity=origin_entity,
            depth=0,
            activation=1.0,
            path=[origin_entity.surface],
            sense_vec=origin_entity.v,
        )
        return ChainResult(
            nodes=[node],
            compound_v=origin_entity.v.copy(),
            origin_v=origin_entity.v.copy(),
            depth_reached=0,
        )

    engine.mda._chain.expand = _origin_only_expand


def _patch_no_oja(engine: MDAEngine) -> None:
    """Freeze all W matrices — update_W becomes a no-op; _w_concept_score returns 0."""

    # Freeze future W updates on entity
    from mda.core import entity as entity_mod
    original_update_W = entity_mod.Entity.update_W

    def _noop_update_W(self, *args, **kwargs):
        pass

    entity_mod.Entity.update_W = _noop_update_W

    # Force W=None on all already-loaded entities so broca falls back to query cosine
    for e in engine.mda.registry.all():
        e.W = None

    # Freeze broca's W scoring
    original_w_concept_score = engine.mda.broca._w_concept_score

    def _zero_w_score(entity, fact_vec):
        return 0.0

    engine.mda.broca._w_concept_score = _zero_w_score

    # Store original so we can warn if called
    engine._ablation_no_oja = True


def apply_ablation(engine: MDAEngine, mode: AblationMode) -> None:
    # Always disable auto-save — ablation runs are stateless, no disk writes
    engine.save = lambda *a, **kw: None
    if mode == "full":
        return
    elif mode == "no_hdr":
        _patch_no_hdr(engine)
    elif mode == "no_graph":
        _patch_no_graph(engine)
    elif mode == "no_oja":
        _patch_no_oja(engine)
    else:
        raise ValueError(f"Unknown ablation mode: {mode!r}")


# ── LLM router (same as benchmark.py) ────────────────────────────────────────

def _llm_complete(model: str, messages: list[dict],
                  provider: str = "ollama", api_key: str = "") -> str:
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")
        key = api_key or __import__("os").environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system, messages=user_msgs,
        )
        raw = resp.content[0].text
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    else:
        import ollama
        resp = ollama.chat(model=model, messages=messages)
        return resp["message"]["content"].strip()


# ── Test case generation (reused from benchmark.py) ──────────────────────────

def generate_test_cases(md_paths: list[str], judge_model: str,
                        questions_per_file: int = 8,
                        provider: str = "ollama", api_key: str = "") -> list[dict]:
    all_cases = []
    for md_path in md_paths:
        content = Path(md_path).read_text(encoding="utf-8")
        if len(content.strip()) < 200:
            console.print(f"[dim]  Skipping {md_path} (too short)[/dim]")
            continue
        label = Path(md_path).stem.upper()
        n_each = max(1, questions_per_file // 4)
        n_fact      = n_each
        n_concept   = n_each
        n_reasoning = n_each
        n_multihop  = questions_per_file - n_each * 3
        prompt = f"""Create a benchmark test set from the document below. Generate exactly {questions_per_file} questions.

You MUST include exactly:
- {n_fact} questions of type "fact": specific number, name, or parameter from the document
- {n_concept} questions of type "concept": what something is or how it works
- {n_reasoning} questions of type "reasoning": how two concepts relate to each other
- {n_multihop} questions of type "multi-hop": chain involving three or more concepts

Rules:
- Questions must be answerable from this document only, not from general knowledge
- Each question needs 2-4 keywords expected in a correct answer
- Each question needs a concise expected_answer (1-2 sentences) taken directly from the document
- Output ONLY a JSON array, nothing else
- Use only straight double quotes in JSON, never apostrophes or smart quotes
- Avoid apostrophes in question and answer text (use "does not" not "doesn't")
- The output array must contain exactly {questions_per_file} objects, no more, no less

[
  {{"type": "fact", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "concept", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "reasoning", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}},
  {{"type": "multi-hop", "question": "...", "keywords": ["kw1", "kw2"], "expected_answer": "..."}}
]

Document:
{content[:6000]}"""

        def _parse(raw: str) -> list:
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            s, e = raw.find("["), raw.rfind("]")
            if s == -1 or e == -1 or e <= s:
                raise ValueError("No JSON array")
            candidate = raw[s:e + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # Repair common issues: smart quotes, unescaped apostrophes in strings
                candidate = candidate.replace("\u2019", "'").replace("\u2018", "'")
                candidate = candidate.replace("\u201c", '"').replace("\u201d", '"')
                # Replace smart quotes with straight quotes outside of JSON strings
                candidate = re.sub(r'[\u2018\u2019]', "'", candidate)
                # Try ast.literal_eval as fallback for simple cases
                try:
                    import ast
                    return ast.literal_eval(candidate)
                except Exception:
                    pass
                # Last resort: extract individual objects with regex
                objects = re.findall(r'\{[^{}]+\}', candidate, re.DOTALL)
                if not objects:
                    raise
                result = []
                for obj in objects:
                    try:
                        result.append(json.loads(obj))
                    except Exception:
                        pass
                if not result:
                    raise
                return result

        try:
            raw = _llm_complete(judge_model, [{"role": "user", "content": prompt}],
                                provider=provider, api_key=api_key)
            try:
                parsed = _parse(raw)
            except (json.JSONDecodeError, ValueError):
                console.print(f"[dim]  Retrying {md_path}...[/dim]")
                raw2 = _llm_complete(judge_model, [{
                    "role": "user",
                    "content": "Output ONLY a valid JSON array. No markdown.\n\n" + prompt,
                }], provider=provider, api_key=api_key)
                parsed = _parse(raw2)

            for i, item in enumerate(parsed[:questions_per_file]):
                all_cases.append({
                    "id": f"{label}{i + 1}",
                    "group": label,
                    "type": item.get("type", "unknown"),
                    "question": item["question"],
                    "expected_keywords": item.get("keywords", []),
                    "expected_answer": item.get("expected_answer", ""),
                })
            console.print(f"[dim]  {min(len(parsed), questions_per_file)} questions from {md_path}[/dim]")
        except Exception as exc:
            console.print(f"[red]  Failed {md_path}: {exc}[/red]")
    return all_cases


# ── MDA build + answer ────────────────────────────────────────────────────────

def build_mda(md_paths: list[str], model: str,
              provider: str = "ollama", api_key: str = "") -> MDAEngine:
    if provider == "anthropic":
        engine = AnthropicEngine(model=model, user_id="ablation", api_key=api_key)
    else:
        engine = MDAEngine(model=model, user_id="ablation")
    # Disable all persistence — ablation runs are stateless
    engine.save        = lambda *a, **kw: None
    engine._auto_load  = lambda *a, **kw: []
    # Clear any state that may have been loaded during __init__
    from mda import MDA
    from mda.inference.reasoning import ReasoningEngine
    engine.mda       = MDA()
    engine._reasoning = ReasoningEngine(engine.mda.encoder, engine.mda.registry)
    total = 0
    for p in md_paths:
        total += engine.load_md(p)
    console.print(f"[dim]    {total} facts loaded[/dim]")
    return engine


def mda_answer(engine: MDAEngine, question: str) -> tuple[str, str]:
    answer = engine.chat(question, lang="en")
    return answer, engine._last_context


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_answer(question: str, answer: str, expected_keywords: list[str],
                 expected_answer: str, judge_model: str,
                 provider: str = "ollama", api_key: str = "",
                 judge_provider: str = "", judge_api_key: str = "") -> dict:
    kw_hits = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
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
        _jp = judge_provider or provider
        _jk = judge_api_key or api_key
        raw = _llm_complete(judge_model, [{"role": "user", "content": prompt}],
                            provider=_jp, api_key=_jk).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = re.search(r"[012]", raw)
        if not m:
            raise ValueError(f"no digit: {raw[:80]!r}")
        score = int(m.group(0))
        reason = raw[:120]
    except Exception as exc:
        score = -1
        reason = f"judge failed: {exc}"

    return {"llm_score": score, "kw_score": kw_score,
            "kw_hits": kw_hits, "kw_total": len(expected_keywords), "reason": reason}


# ── Single config run ─────────────────────────────────────────────────────────

def run_config(md_paths: list[str], model: str, judge_model: str,
               mode: AblationMode, test_cases: list[dict],
               provider: str = "ollama", api_key: str = "",
               judge_provider: str = "", judge_api_key: str = "") -> list[dict]:
    label = ABLATION_LABELS[mode]
    console.print(f"\n[bold yellow]Config: {label}[/bold yellow]")
    engine = build_mda(md_paths, model, provider=provider, api_key=api_key)
    apply_ablation(engine, mode)

    results = []
    for tc in test_cases:
        t0 = time.time()
        answer, ctx = mda_answer(engine, tc["question"])
        elapsed = round(time.time() - t0, 2)

        j = judge_answer(
            tc["question"], answer,
            tc["expected_keywords"], tc.get("expected_answer", ""),
            judge_model, provider=provider, api_key=api_key,
            judge_provider=judge_provider, judge_api_key=judge_api_key,
        )

        results.append({
            "id": tc["id"],
            "group": tc["group"],
            "type": tc.get("type", "unknown"),
            "question": tc["question"],
            "mode": mode,
            "answer": answer,
            "context_len": len(ctx),
            "time_s": elapsed,
            **j,
        })

        score_str = str(j["llm_score"]) if j["llm_score"] >= 0 else "ERR"
        console.print(
            f"  [dim][{tc['id']}][/dim] score={score_str}  "
            f"ctx={len(ctx)}c  {tc['question'][:55]}"
        )

    return results


# ── Report ────────────────────────────────────────────────────────────────────

def _accuracy(results: list[dict]) -> float:
    valid = [r for r in results if r["llm_score"] >= 0]
    if not valid:
        return 0.0
    return round(sum(r["llm_score"] for r in valid) / (len(valid) * 2) * 100, 1)


def _by_type(results: list[dict]) -> dict[str, float]:
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for r in results:
        buckets[r["type"]].append(r)
    return {t: _accuracy(rs) for t, rs in sorted(buckets.items())}


def print_ablation_report(all_results: dict[str, list[dict]]) -> None:
    console.rule("[bold]Ablation Study Results[/bold]")

    # Overall accuracy table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Config", width=24)
    table.add_column("Overall %", justify="right", width=10)
    table.add_column("vs Full MDA", justify="right", width=12)
    table.add_column("Avg ctx len", justify="right", width=12)

    full_acc = _accuracy(all_results.get("full", []))
    for mode in ("full", "no_hdr", "no_graph", "no_oja"):
        if mode not in all_results:
            continue
        results = all_results[mode]
        acc = _accuracy(results)
        delta = acc - full_acc
        delta_str = (f"[green]+{delta:.1f}[/green]" if delta > 0
                     else f"[red]{delta:.1f}[/red]" if delta < 0 else "—")
        avg_ctx = round(sum(r["context_len"] for r in results) / len(results)) if results else 0
        table.add_row(ABLATION_LABELS[mode], f"{acc:.1f}%", delta_str, f"{avg_ctx:,}c")

    console.print(table)

    # Per question-type breakdown
    console.print("\n[bold]Per question type:[/bold]")
    type_table = Table(show_header=True, header_style="bold cyan")
    type_table.add_column("Type", width=12)
    for mode in ("full", "no_hdr", "no_graph", "no_oja"):
        if mode in all_results:
            type_table.add_column(ABLATION_LABELS[mode][:16], justify="right", width=16)

    all_types = set()
    type_accs: dict[str, dict[str, float]] = {}
    for mode, results in all_results.items():
        bt = _by_type(results)
        type_accs[mode] = bt
        all_types.update(bt.keys())

    for t in sorted(all_types):
        row = [t]
        for mode in ("full", "no_hdr", "no_graph", "no_oja"):
            if mode not in all_results:
                continue
            acc = type_accs.get(mode, {}).get(t, None)
            row.append(f"{acc:.1f}%" if acc is not None else "—")
        type_table.add_row(*row)

    console.print(type_table)

    # Contribution summary
    console.print("\n[bold]Component contribution (drop from Full MDA):[/bold]")
    for mode in ("no_hdr", "no_graph", "no_oja"):
        if mode not in all_results:
            continue
        drop = full_acc - _accuracy(all_results[mode])
        label = ABLATION_LABELS[mode]
        if drop > 5:
            color = "red"
        elif drop > 0:
            color = "yellow"
        else:
            color = "green"
        console.print(f"  {label:<22} → [{color}]{drop:+.1f}pp[/{color}]")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_results(all_results: dict[str, list[dict]], output_dir: str, model: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = model.replace(":", "-").replace("/", "-")
    path = out / f"ablation_{slug}_{ts}.json"
    path.write_text(
        json.dumps({"configs": ABLATION_LABELS, "results": all_results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Results saved → {path}[/dim]")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MDA Ablation Study")
    parser.add_argument("--md", nargs="+", required=True,
                        help="Knowledge base markdown files")
    parser.add_argument("--model", default="qwen3:4b",
                        help="LLM model for answers")
    parser.add_argument("--judge", default=None,
                        help="Judge model (default: same as --model)")
    parser.add_argument("--questions", type=int, default=8,
                        help="Questions per md file (default: 8)")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "anthropic"])
    parser.add_argument("--api-key", default="",
                        help="API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--judge-provider", default="",
                        choices=["", "ollama", "anthropic"],
                        help="Provider for judge model (default: same as --provider)")
    parser.add_argument("--judge-api-key", default="",
                        help="API key for judge (default: same as --api-key)")
    parser.add_argument("--output", default="results/ablation",
                        help="Output directory (default: results/ablation)")
    parser.add_argument("--configs", nargs="+",
                        choices=["full", "no_hdr", "no_graph", "no_oja"],
                        default=["full", "no_hdr", "no_graph", "no_oja"],
                        help="Which ablation configs to run (default: all)")
    args = parser.parse_args()

    judge_model = args.judge or args.model

    judge_provider = args.judge_provider or args.provider
    judge_api_key  = args.judge_api_key  or args.api_key

    console.rule("[bold]MDA Ablation Study[/bold]")
    console.print(f"[dim]  model: {args.model}  judge: {judge_model}  "
                  f"provider: {args.provider}  judge-provider: {judge_provider}[/dim]")
    console.print(f"[dim]  configs: {args.configs}[/dim]\n")

    console.print("[bold yellow]Generating test cases...[/bold yellow]")
    test_cases = generate_test_cases(
        args.md, judge_model, args.questions,
        provider=judge_provider, api_key=judge_api_key,
    )
    if not test_cases:
        console.print("[red]No test cases generated. Exiting.[/red]")
        return
    console.print(f"[bold]  {len(test_cases)} questions ready[/bold]\n")

    all_results: dict[str, list[dict]] = {}
    for mode in args.configs:
        all_results[mode] = run_config(
            args.md, args.model, judge_model, mode, test_cases,
            provider=args.provider, api_key=args.api_key,
            judge_provider=judge_provider, judge_api_key=judge_api_key,
        )

    print_ablation_report(all_results)
    save_results(all_results, args.output, args.model)


if __name__ == "__main__":
    main()