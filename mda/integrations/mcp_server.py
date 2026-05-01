"""
MDA MCP Server — expose MDA as a Model Context Protocol (MCP) server.

MDA is used as a memory/context layer only.
The client's own model handles generation — no LLM call is made here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP("MDA Memory Server", stateless_http=True)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

_PROVIDER = os.environ.get("MDA_PROVIDER", "ollama")
_BASE_URL  = os.environ.get("MDA_BASE_URL", "")
_USER_ID   = os.environ.get("MDA_USER_ID", "default")
_LANG      = os.environ.get("MDA_LANG", "en")
_MODEL     = os.environ.get("MDA_MODEL", "default")

# Mutable active selection — changed by mda_switch_model
_active_model:    str = _MODEL
_active_provider: str = _PROVIDER

# ---------------------------------------------------------------------------
# Engine registry — one instance per (provider, model) pair
# ---------------------------------------------------------------------------

_engines: dict[str, Any] = {}  # key: "provider:model"
_batch_engine: Any = None       # singleton MDABatchEngine


def _get_batch_engine() -> Any:
    """Return (or lazily create) the singleton MDABatchEngine."""
    global _batch_engine
    if _batch_engine is None:
        from mda.integrations.engine import MDABatchEngine
        _batch_engine = MDABatchEngine(
            model=_active_model,
            user_id=_USER_ID,
        )
    return _batch_engine


def _current_engine() -> Any:
    """Return the engine for the currently active (provider, model) pair."""
    return _get_engine(_active_model, _active_provider)


def _get_engine(model: str, provider: str | None = None) -> Any:
    """Return a cached MDAEngine for the given (provider, model) pair.

    On first creation:
      1. Calls _auto_load() to restore any persisted session.
      2. Discovers and loads all .md files via _discover_md_files() +
         _load_md_file().
    """
    from mda.integrations.engine import MDAEngine, AnthropicEngine

    prov = provider or _PROVIDER
    key  = f"{prov}:{model}"

    if key not in _engines:
        if prov == "anthropic":
            engine: Any = AnthropicEngine(
                model=model,
                user_id=_USER_ID,
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        else:
            engine = MDAEngine(
                model=model,
                user_id=_USER_ID,
                provider=prov,
                base_url=_BASE_URL,
            )

        # Restore persisted memory
        engine._auto_load()

        # Load all discovered .md files
        base = Path(__file__).parent.parent.parent
        for md_path in engine._discover_md_files(base):
            try:
                engine._load_md_file(md_path)
            except Exception:
                pass

        _engines[key] = engine

    return _engines[key]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def mda_context(query: str) -> str:
    """Build and return the raw MDA context string for a query.

    This is the RAG equivalent: the caller receives the context and injects it
    into their own LLM prompt. No LLM call is made inside MDA.
    """
    engine   = _current_engine()
    en_query = engine.mda._translator.to_english(query)
    context  = engine._build_context(en_query)
    return context or ""


@mcp.tool()
def mda_learn(text: str) -> str:
    """Store free-form text into MDA memory."""
    engine = _current_engine()
    engine.mda.learn(text)
    return f"Learned: {text[:80]}{'...' if len(text) > 80 else ''}"


@mcp.tool()
def mda_teach(surface: str, facts: list[str]) -> str:
    """Store structured facts for a named entity (category='custom')."""
    engine = _current_engine()
    engine.mda.teach(surface, facts, category="custom")
    return f"Taught {len(facts)} fact(s) for entity '{surface}'"


@mcp.tool()
def mda_load_md(path: str) -> str:
    """Load a markdown file into MDA memory."""
    engine = _current_engine()
    p = Path(path.strip().strip("'\""))
    if not p.exists():
        return f"File not found: {path}"
    count = engine._load_md_file(p)
    return f"Loaded {count} paragraph(s) from {p.name}"


@mcp.tool()
def mda_load_file(path: str) -> str:
    """Load any supported file (.py, .md, .rs, .ts, .go, .json, .yaml, .txt …) into MDA memory using the Loader pipeline."""
    engine = _current_engine()
    p = Path(path.strip().strip("'\""))
    if not p.exists():
        return f"File not found: {path}"
    count = engine.loader.load_file(str(p))
    return f"Loaded {count} fact(s) from {p.name}"


@mcp.tool()
def mda_load_dir(directory: str, extensions: list[str] = None, recursive: bool = True) -> dict:
    """Load all supported files in a directory into MDA memory using the Loader pipeline.

    Args:
        directory:  Root directory to scan.
        extensions: Optional extension whitelist e.g. [".py", ".md"]. Defaults to all supported.
        recursive:  Whether to descend into subdirectories (default True).
    """
    engine = _current_engine()
    d = Path(directory.strip().strip("'\""))
    if not d.exists():
        return {"error": f"Directory not found: {directory}"}
    outcomes = engine.loader.load_dir(str(d), extensions=extensions, recursive=recursive)
    total_facts = sum(outcomes.values())
    return {
        "files_processed": len(outcomes),
        "total_facts": total_facts,
        "details": {str(Path(p).name): c for p, c in outcomes.items() if c > 0},
    }


@mcp.tool()
def mda_switch_model(model: str, provider: str = "") -> str:
    """Switch the active model (and optionally provider) used by all MDA tools.

    The engine for the new (provider, model) pair is initialised immediately
    if it has not been used before (session loaded, .md files ingested).
    The previous engine stays cached and can be restored by calling this tool
    again.

    Args:
        model:    Model name, e.g. "qwen3:4b", "claude-haiku-4-5-20251001",
                  or a llama.cpp model filename.
        provider: One of "ollama", "llama_cpp", "anthropic".
                  Defaults to the current active provider if omitted.
    """
    global _active_model, _active_provider
    prev_key          = f"{_active_provider}:{_active_model}"
    _active_provider  = provider or _active_provider
    _active_model     = model
    _get_engine(_active_model, _active_provider)  # eagerly init / warm cache
    new_key = f"{_active_provider}:{_active_model}"
    return f"Switched {prev_key} → {new_key}"


@mcp.tool()
def mda_list_engines() -> dict:
    """List all cached engines and show which one is currently active."""
    return {
        "active": f"{_active_provider}:{_active_model}",
        "cached": list(_engines.keys()),
    }


@mcp.tool()
def mda_batch_context(queries: list[str]) -> list[str]:
    """Build MDA context strings for multiple queries in a single GPU pass.

    Returns one context string per query in the same order.
    Use this instead of calling mda_context() N times — the entity matrix
    is traversed once for the whole batch, making it significantly faster
    for 5+ concurrent queries.
    """
    engine = _get_batch_engine()
    return engine.build_context_batch(queries)


@mcp.tool()
def mda_batch_learn(texts: list[str]) -> str:
    """Store multiple texts into MDA memory and rebuild the entity matrix once.

    More efficient than calling mda_learn() N times because the entity
    matrix rebuild (GPU pass) happens only once after all texts are ingested.
    """
    engine = _get_batch_engine()
    engine.learn_batch(texts)
    return f"Learned {len(texts)} text(s) in batch"


@mcp.tool()
def mda_batch_save() -> str:
    """Save the batch engine session to disk."""
    if _batch_engine is None:
        return "Batch engine not initialised"
    try:
        _batch_engine.save()
        return "Batch engine saved"
    except Exception as exc:
        return f"Batch save error: {exc}"


@mcp.tool()
def mda_save() -> str:
    """Save all active engine sessions to disk."""
    saved: list[str] = []
    for key, engine in _engines.items():
        try:
            engine.save()
            saved.append(key)
        except Exception as exc:
            saved.append(f"{key} (error: {exc})")
    return f"Saved: {', '.join(saved)}" if saved else "No active engines to save"


@mcp.tool()
def mda_stats() -> dict:
    """Return memory statistics for all active engines.

    Returns entity count, synapse count, fact count, active engine keys,
    and the configured user_id.
    """
    engine_stats: dict[str, dict] = {}
    for key, engine in _engines.items():
        entities   = engine.mda.registry._entities
        n_entities = len(entities)
        n_synapses = sum(len(e.synapses) for e in entities.values())
        n_facts    = sum(
            len(facts)
            for facts in engine.mda.broca._entity_facts.values()
        )
        engine_stats[key] = {
            "entity_count":  n_entities,
            "synapse_count": n_synapses,
            "fact_count":    n_facts,
        }
    batch_stats: dict | None = None
    if _batch_engine is not None:
        b_entities  = _batch_engine.mda.registry._entities
        batch_stats = {
            "entity_count":  len(b_entities),
            "synapse_count": sum(len(e.synapses) for e in b_entities.values()),
            "fact_count":    sum(
                len(f) for f in _batch_engine.mda.broca._entity_facts.values()
            ),
        }

    return {
        "engines":      engine_stats,
        "batch_engine": batch_stats,
        "active_keys":  list(_engines.keys()),
        "user_id":      _USER_ID,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MDA MCP Server")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio", "sse"],
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="SSE bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8766,
        help="SSE bind port (default: 8766)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        app = mcp.streamable_http_app()
        cors_app = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        uvicorn.run(cors_app, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
