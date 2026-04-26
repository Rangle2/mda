from __future__ import annotations

import argparse
from pathlib import Path
import dotenv
dotenv.load_dotenv()
import requests
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.text import Text

from mda.integrations.engine import MDAEngine, AnthropicEngine


# ---------------------------------------------------------------------------
# Ollama model picker
# ---------------------------------------------------------------------------

_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _fetch_ollama_models() -> list[str]:
    """Return list of locally available Ollama model names, or [] on failure."""
    try:
        resp = requests.get(_OLLAMA_TAGS_URL, timeout=4)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def pick_model(console: Console, default: str) -> str:
    """Display available Ollama models and let the user pick one.
    Returns the chosen model name, or `default` if Ollama is unreachable.
    """
    models = _fetch_ollama_models()
    if not models:
        print_sys(console, f"Ollama not reachable — using default model: {default}")
        return default

    console.print()
    console.print(Text("  Available models:", style="#888888"))
    for i, name in enumerate(models, 1):
        marker = " *" if name == default else ""
        console.print(Text(f"  [{i}] {name}{marker}", style="#aaaaaa"))
    console.print(Text("  [0] enter model name manually", style="#666666"))
    console.print()

    pt_style = Style.from_dict({"prompt": "#555577", "": "#cccccc"})
    session  = PromptSession()
    while True:
        try:
            with patch_stdout():
                raw = session.prompt(f"  select model [1-{len(models)}] or 0 > ",
                                     style=pt_style)
            raw = raw.strip()
            if raw == "0":
                with patch_stdout():
                    name = session.prompt("  model name > ", style=pt_style)
                name = name.strip()
                return name if name else default
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print_sys(console, f"enter a number between 0 and {len(models)}")
        except (ValueError, KeyboardInterrupt):
            continue
        except EOFError:
            return default


# ---------------------------------------------------------------------------
# Logo
# ---------------------------------------------------------------------------

LOGO = r"""
 __  __  ____   ___
|  \/  ||  _ \ / _ \
| |\/| || | | | |_| |
| |  | || |_| |  _ |
|_|  |_||____/ |_| |_|
"""


def print_logo(console: Console) -> None:
    for line in LOGO.splitlines():
        console.print(Text(line, style="medium_purple"))
    console.print()
    print_sys(console, "/help  /learn  /teach  /model  /lang  /smart")
    print_sys(console, "/save  /load  /loadmd  /debug  /prune  /clear  /exit")
    console.print()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_sep(console: Console) -> None:
    console.print(Text("  " + "-" * 50, style="#2a2a2a"))


def print_status(console: Console, bridge: MDAEngine | None, lang: str) -> None:
    if bridge is None:
        return
    n_e = len(bridge.mda.registry._entities)
    n_s = sum(len(e.synapses) for e in bridge.mda.registry._entities.values())
    parts = Path(bridge._memory_base).parts
    mem_short = f".memory/{parts[-1]}" if parts else ".memory/default"

    t = Text()
    t.append("  ")
    t.append(bridge.model, style="medium_purple")
    t.append("  ", style="")
    t.append(lang, style="#666666")
    t.append("  entity:", style="#444444")
    t.append(str(n_e), style="#888888")
    t.append("  syn:", style="#444444")
    t.append(str(n_s), style="#888888")
    t.append("  ", style="")
    t.append(mem_short, style="#555577")
    console.print(t)


def print_user(console: Console, text: str) -> None:
    t = Text()
    t.append("  you  ", style="#444466")
    t.append(text, style="#c0c0c0")
    console.print(t)
    console.print()


def print_model(console: Console, model_name: str, text: str) -> None:
    console.print()
    first = True
    for line in text.strip().splitlines():
        t = Text()
        if first:
            t.append(f"  {model_name[:12]:<12}  ", style="#7c6af7")
            first = False
        else:
            t.append("  " + " " * min(len(model_name), 12) + "  ", style="")
        t.append(line, style="#f0f0f0")
        console.print(t)
    console.print()


def print_thinking(console: Console, text: str) -> None:
    if not text.strip():
        return
    lines = text.strip().splitlines()
    console.print(Text("  <thinking>", style="#7060aa"))
    for line in lines[:5]:
        console.print(Text("    " + line[:90], style="#9080cc italic"))
    if len(lines) > 5:
        console.print(Text(f"    ... ({len(lines) - 5} more lines)", style="#7060aa"))
    console.print(Text("  </thinking>", style="#7060aa"))
    console.print()


def print_sys(console: Console, text: str) -> None:
    for line in text.splitlines():
        t = Text()
        t.append("  [·] ", style="#555555")
        t.append(line, style="#bbbbbb")
        console.print(t)


def print_concepts(console: Console, bridge: MDAEngine, user_text: str) -> None:
    """Always-visible MDA layer: entities activated + top context lines."""
    en_q = bridge._last_en_query or user_text

    # Entities MDA activated
    entities = bridge.mda._find_entities_from_text(en_q)
    if entities:
        t = Text()
        t.append("  entities  ", style="#3a8a8a")
        t.append("  ".join(e.surface for e in entities[:8]), style="#5fafaf")
        console.print(t)

    # Top synapses for first entity
    if entities:
        e0 = entities[0]
        strong = sorted(e0.synapses.values(), key=lambda s: -s.strength)[:4]
        if strong:
            t = Text()
            t.append("  related   ", style="#3a6a6a")
            parts = []
            for syn in strong:
                other = bridge.mda.registry.get_by_id(syn.target_id)
                if other:
                    parts.append(f"{other.surface}:{syn.strength:.2f}")
            t.append("  ".join(parts), style="#4a9090")
            console.print(t)

    # Context lines MDA fed to LLM
    ctx = bridge._build_context(user_text)
    ctx_lines = [l for l in ctx.splitlines() if l.strip()][:3]
    if ctx_lines:
        t = Text()
        t.append("  context   ", style="#3a5a7a")
        # strip [MEMORY] prefix for display
        first_line = ctx_lines[0].replace("[MEMORY] ", "")
        t.append(first_line[:80], style="#6a8aaa")
        console.print(t)
        for line in ctx_lines[1:]:
            t2 = Text()
            t2.append("             ", style="")
            t2.append(line.replace("[MEMORY] ", "")[:80], style="#6a8aaa")
            console.print(t2)

    console.print()


def print_debug(console: Console, bridge: MDAEngine, user_text: str) -> None:
    """Verbose debug — shown when debug mode is on."""
    en_q = bridge._last_en_query or ""
    rows: list[tuple[str, str]] = [
        ("en_query",    en_q[:90]),
        ("en_response", (bridge._last_en_response or "")[:90]),
    ]
    ctx = bridge._build_context(user_text)
    for i, line in enumerate(ctx.splitlines()[:5]):
        rows.append((f"ctx[{i}]", line[:90]))

    for key, val in rows:
        t = Text()
        t.append(f"  {key:<14}", style="#555555")
        t.append(val, style="#999999")
        console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def handle_cmd(raw: str, console: Console, bridge: MDAEngine,
               lang_ref: dict, last_input_ref: dict,
               debug_ref: dict) -> None:
    parts = raw.split(None, 2)
    name  = parts[0].lstrip("/").lower()

    if name == "help":
        lines = [
            "/learn <text>            learn text into MDA",
            "/teach <entity> <fact>   teach entity a fact",
            "/model <name>            switch model",
            "/lang en|tr              switch language",
            "/smart                   toggle smart filter",
            "/debug                   toggle debug mode (verbose per turn)",
            "/save                    save session to .memory/",
            "/load                    reload session from .memory/",
            "/loadmd [path]           load markdown file as facts",
            "/prune                   remove weak entities from registry",
            "/clear                   clear screen",
            "/exit /quit              save and exit",
            "/help                    this message",
        ]
        for line in lines:
            print_sys(console, line)

    elif name == "learn":
        if len(parts) < 2:
            print_sys(console, "usage: /learn <text>")
            return
        body = raw.split(None, 1)[1]
        bridge.learn(body)
        print_sys(console, f"learned · {body[:60]}")

    elif name == "teach":
        if len(parts) < 3:
            print_sys(console, "usage: /teach <entity> <fact>")
            return
        bridge.teach(parts[1], [parts[2]])
        print_sys(console, f"taught {parts[1]!r} · {parts[2][:50]}")

    elif name == "model":
        if len(parts) < 2:
            print_sys(console, "usage: /model <name>")
            return
        msgs = bridge.switch_model(parts[1])
        print_sys(console, f"model -> {parts[1]}")
        for msg in msgs:
            print_sys(console, msg)

    elif name == "lang":
        if len(parts) < 2:
            print_sys(console, "usage: /lang en|tr")
            return
        lang_ref["lang"] = parts[1].lower()
        print_sys(console, f"language -> {lang_ref['lang']}")

    elif name == "smart":
        bridge.smart_filter = not bridge.smart_filter
        print_sys(console, f"smart filter -> {'on' if bridge.smart_filter else 'off'}")

    elif name == "debug":
        debug_ref["on"] = not debug_ref["on"]
        state = "ON" if debug_ref["on"] else "OFF"
        print_sys(console, f"debug mode -> {state}")

    elif name == "save":
        try:
            meta = bridge.save()
            print_sys(console,
                f"saved · {meta.get('session_id', '?')[:8]} "
                f"· turns:{meta.get('turn_count', '?')} "
                f"· {meta.get('updated_at', '?')[:19]}"
            )
        except Exception as exc:
            print_sys(console, f"save failed: {exc}")

    elif name == "load":
        try:
            msgs = bridge._auto_load()
            for msg in msgs:
                print_sys(console, msg)
        except Exception as exc:
            print_sys(console, f"load failed: {exc}")

    elif name == "loadmd":
        path = parts[1] if len(parts) >= 2 else None
        try:
            count = bridge.load_md(path)
            print_sys(console, f"loaded {count} facts" + (f" from {path}" if path else ""))
        except Exception as exc:
            print_sys(console, f"loadmd failed: {exc}")

    elif name == "prune":
        pruned    = bridge.mda.registry.prune()
        remaining = len(list(bridge.mda.registry.all()))
        print_sys(console, f"pruned {pruned} weak entities · registry: {remaining} remaining")

    elif name == "clear":
        import os, sys
        os.system("cls" if sys.platform == "win32" else "clear")

    elif name in ("exit", "quit"):
        print_sys(console, "saving...")
        try:
            bridge.save()
        except Exception as exc:
            print_sys(console, f"save failed: {exc}")
        print_sys(console, "goodbye")
        raise SystemExit(0)

    else:
        print_sys(console, f"unknown: /{name}  (try /help)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(model: str, lang: str, smart: bool, knowledge: str | None,
        max_entities: int | None, md: str | None, user_id: str,
        provider: str = "ollama", api_key: str = "") -> None:
    console = Console(highlight=False)

    print_logo(console)

    if provider == "anthropic":
        print_sys(console, f"provider: anthropic  model: {model}")
        bridge = AnthropicEngine(
            model=model,
            knowledge_path=knowledge,
            max_entities=max_entities,
            smart_filter=smart,
            user_id=user_id,
            api_key=api_key,
        )
    else:
        model = pick_model(console, model)
        bridge = MDAEngine(
            model=model,
            knowledge_path=knowledge,
            max_entities=max_entities,
            smart_filter=smart,
            user_id=user_id,
        )

    # Auto-load previous session
    msgs = bridge._auto_load()
    for msg in msgs:
        print_sys(console, msg)

    # Auto-discover and load .md files — per-file progress
    base = Path(__file__).parent.parent.parent
    md_files = bridge._discover_md_files(base)
    if md_files:
        for f in md_files:
            print_sys(console, f"loading {f.name} ...")
            try:
                count = bridge._load_md_file(f)
                print_sys(console, f"  {f.name} -> {count} facts")
            except Exception as exc:
                print_sys(console, f"  {f.name} failed: {exc}")

    # Optional --md flag
    if md:
        p = Path(md)
        print_sys(console, f"loading {p.name} ...")
        try:
            count = bridge._load_md_file(p)
            print_sys(console, f"  {p.name} -> {count} facts")
        except Exception as exc:
            print_sys(console, f"  {p.name} failed: {exc}")

    lang_ref       = {"lang": lang}
    last_input_ref = {"text": ""}
    debug_ref      = {"on": False}

    session  = PromptSession()
    pt_style = Style.from_dict({
        "prompt": "#555577",
        "":       "#cccccc",
    })

    while True:
        try:
            print_sep(console)
            print_status(console, bridge, lang_ref["lang"])
            print_sep(console)

            with patch_stdout():
                raw = session.prompt("  > ", style=pt_style)

            raw = raw.strip()
            if not raw:
                continue

            if raw.startswith("/"):
                handle_cmd(raw, console, bridge, lang_ref, last_input_ref, debug_ref)
            else:
                last_input_ref["text"] = raw
                print_user(console, raw)

                print_sys(console, f"··· {bridge.model}")
                response = bridge.chat(raw, lang=lang_ref["lang"])
                thinking = bridge._last_thinking

                if thinking:
                    print_thinking(console, thinking)

                print_model(console, bridge.model, response)

                # Always show MDA concepts layer
                print_concepts(console, bridge, raw)

                # Extra verbose info when debug mode is on
                if debug_ref["on"]:
                    print_debug(console, bridge, raw)

        except KeyboardInterrupt:
            continue
        except EOFError:
            print_sys(console, "saving...")
            try:
                bridge.save()
            except Exception as exc:
                print_sys(console, f"save failed: {exc}")
            print_sys(console, "goodbye")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mda",
                                description="MDA · memory-augmented LLM shell")
    p.add_argument("--model",        default="qwen3:4b")
    p.add_argument("--lang",         default="en", choices=["en", "tr"])
    p.add_argument("--smart",        action="store_true")
    p.add_argument("--knowledge",    default=None, metavar="PATH")
    p.add_argument("--max-entities", default=None, type=int, metavar="N")
    p.add_argument("--md",           default=None, metavar="PATH")
    p.add_argument("--user-id",      default="default", metavar="ID")
    p.add_argument("--provider",     default="ollama", choices=["ollama", "anthropic"],
                   help="LLM provider (default: ollama)")
    p.add_argument("--api-key",      default="", metavar="KEY",
                   help="API key (or set ANTHROPIC_API_KEY env var)")
    return p.parse_args()


def main() -> None:
    a = _args()
    run(
        model=a.model, lang=a.lang, smart=a.smart,
        knowledge=a.knowledge, max_entities=a.max_entities,
        md=a.md, user_id=a.user_id,
        provider=a.provider, api_key=a.api_key,
    )


if __name__ == "__main__":
    main()
