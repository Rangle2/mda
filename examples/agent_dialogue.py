"""
examples/agent_dialogue.py
--------------------------
Two agents with zero prior knowledge collaborate on a topic using MDA
as their shared associative memory.

Architecture
------------
  Explorer   ──────────────────────────────────────────────────────────────
                  ↓  learn()          ↑  _build_context()
             ┌──────────────────────────────────────────┐
             │           Shared MDAEngine               │
             │  entity store · synapse graph · broca    │
             └──────────────────────────────────────────┘
                  ↓  learn()          ↑  _build_context()
  Synthesizer ─────────────────────────────────────────────────────────────

Each turn:
  1. Active agent calls _build_context(last_message) to retrieve what MDA
     has learned so far about the topic.
  2. That context is injected into the LLM prompt as [SHARED MEMORY].
  3. The agent generates a response (2-3 sentences, no prior knowledge).
  4. The response is learned into MDA via engine.learn(), making it
     available to both agents on the next turn.

Usage
-----
  python examples/agent_dialogue.py                        # default topic
  python examples/agent_dialogue.py --topic "dark matter"
  python examples/agent_dialogue.py --turns 8 --model qwen2.5:14b
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import NamedTuple

import requests

# Allow running from repo root without installing the package.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from mda.integrations.engine import MDAEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:9b"
DEFAULT_TOPIC = "How does the brain form long-term memories at the synaptic level?"
DEFAULT_TURNS = 8

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

class Agent(NamedTuple):
    name: str
    system: str
    color: str  # ANSI escape for terminal colour


AGENTS: list[Agent] = [
    Agent(
        name="Explorer",
        system=(
            "You are Explorer, a curious research agent. "
            "Your role is to surface hypotheses, propose mechanisms, and ask "
            "probing questions that push the dialogue forward. "
            "You have NO prior knowledge of the topic — "
            "build your understanding step-by-step from the [SHARED MEMORY] "
            "provided each turn. "
            "Keep every response to 2-3 sentences maximum."
        ),
        color="\033[36m",   # cyan
    ),
    Agent(
        name="Synthesizer",
        system=(
            "You are Synthesizer, a critical-reasoning agent. "
            "Your role is to evaluate Explorer's claims, connect emerging ideas, "
            "and incrementally consolidate understanding toward a conclusion. "
            "You have NO prior knowledge of the topic — "
            "build your understanding step-by-step from the [SHARED MEMORY] "
            "provided each turn. "
            "Keep every response to 2-3 sentences maximum."
        ),
        color="\033[33m",   # yellow
    ),
]

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_ollama(model: str, system: str, user_content: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.7},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot reach Ollama at {OLLAMA_URL}. Is it running?")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] LLM call failed: {exc}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)

def print_header(topic: str, model: str, turns: int) -> None:
    sep = "=" * 68
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}  MDA Agent Dialogue{RESET}")
    print(f"  Topic  : {topic}")
    print(f"  Model  : {model}")
    print(f"  Turns  : {turns}")
    print(f"{BOLD}{sep}{RESET}\n")

def print_turn(turn: int, agent: Agent, context: str, response: str) -> None:
    print(f"{DIM}-- Turn {turn} ----------------------------------------------------------{RESET}")
    print(f"{agent.color}{BOLD}{agent.name}{RESET}")

    if context:
        print(f"{DIM}  [MDA context]{RESET}")
        for line in context.splitlines():
            print(f"{DIM}  | {line}{RESET}")

    print(_wrap(response, indent="  "))
    print()

def print_footer(engine: MDAEngine) -> None:
    entities  = engine.mda.registry.count()
    facts     = sum(len(f) for f in engine.mda.broca._entity_facts.values())
    synapses  = sum(len(e.synapses) for e in engine.mda.registry._entities.values())
    events    = len(engine.mda._event_store)
    sep = "=" * 68
    print(f"{BOLD}{sep}{RESET}")
    print(f"{BOLD}  MDA state after dialogue{RESET}")
    print(f"  Entities : {entities}")
    print(f"  Synapses : {synapses}")
    print(f"  Facts    : {facts}")
    print(f"  Events   : {events}")
    print(f"{BOLD}{sep}{RESET}\n")

# ---------------------------------------------------------------------------
# Dialogue loop
# ---------------------------------------------------------------------------

def run_dialogue(topic: str, model: str, turns: int) -> None:
    engine = MDAEngine(model=model)

    print_header(topic, model, turns)

    # Seed the conversation with the topic so both agents start from the same
    # anchor — MDA gets one entity/fact for the topic phrase itself.
    engine.learn(f"Topic under investigation: {topic}")

    current_message = (
        f"We are going to explore this topic together: {topic}\n"
        f"Neither of us has any background knowledge about it. "
        f"Please share your first hypothesis or question."
    )

    for turn in range(1, turns + 1):
        agent = AGENTS[(turn - 1) % 2]

        # 1. Retrieve what MDA knows so far
        context = engine._build_context(current_message)

        # 2. Build LLM prompt — inject shared memory if available
        if context:
            user_content = (
                f"[SHARED MEMORY — what we have established so far]\n"
                f"{context}\n\n"
                f"[PARTNER'S LAST MESSAGE]\n"
                f"{current_message}"
            )
        else:
            user_content = current_message

        # 3. Generate response
        response = call_ollama(model, agent.system, user_content)

        # 4. Learn response into shared MDA — available to both agents next turn
        engine.learn(f"{agent.name}: {response}")

        print_turn(turn, agent, context, response)

        current_message = response

    print_footer(engine)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDA two-agent dialogue demo")
    parser.add_argument("--topic",  default=DEFAULT_TOPIC,
                        help="Topic for agents to explore")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help="Ollama model name")
    parser.add_argument("--turns",  type=int, default=DEFAULT_TURNS,
                        help="Total number of agent turns")
    args = parser.parse_args()

    run_dialogue(topic=args.topic, model=args.model, turns=args.turns)
