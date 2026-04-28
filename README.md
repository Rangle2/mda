# Modular Dynamic Architecture (MDA)

**Online associative memory for LLMs. Learns during inference. No backpropagation.**

[![PyPI](https://img.shields.io/pypi/v/mda-memory.svg)](https://pypi.org/project/mda-memory/)
[![License: SSPL](https://img.shields.io/badge/License-SSPL-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What is MDA?

Large language models can reason but cannot remember. RAG partially addresses this but cannot update during a conversation or learn from it.

MDA fills precisely these gaps.

It encodes knowledge as **512-dimensional Holographic Distributed Representations (HDRs)**, connects concepts through a **sparse synapse graph**, and retrieves context by activating entity networks — not by text-chunk similarity search. New knowledge is integrated immediately, without rebuilding any index.

**MDA is not a RAG replacement. It is the persistent learning layer that RAG and LLMs are missing.**

---

## Key Properties

- **Token-free** — no tokenizer, no vocabulary
- **Attention-free** — no transformer encoder required
- **Online learning** — learns during inference via the Oja rule
- **No catastrophic forgetting** — entities are independent; new knowledge never overwrites old
- **CPU-first** — runs on numpy; GPU acceleration via PyTorch when available
- **Model-agnostic** — works with Ollama, OpenAI, Anthropic, llama.cpp, or any LLM

---

## Benchmark Results

Evaluated against a strong RAG baseline (bge-large-en-v1.5 + ChromaDB, top-6 retrieval) across 80 questions spanning 8 cognitive categories:

| Category | RAG | MDA | Δ |
|---|---|---|---|
| ATOMIC_RECALL | 100% | 85% | −15% |
| MULTI_HOP | 90% | 90% | 0% |
| CROSS_DOCUMENT | 80% | 70% | −10% |
| REASONING | 70% | 90% | **+20%** |
| INCREMENTAL_LEARNING | 0% | 60% | **+60%** |
| NOISE_RESISTANCE | 100% | 100% | 0% |
| MEMORY_COMPRESSION | 90% | 70% | −20% |
| BOUNDARY | 100% | 100% | 0% |
| **OVERALL** | **78.8%** | **83.1%** | **+4.3%** |

MDA uses **3.1× less context per query** than RAG while achieving higher overall accuracy.

**Long-context retention (200 turns):** RAG 0% — MDA 92%.

### Ablation Study

| Config | Run 1 | Run 2 | Run 3 | Avg | Δ vs Full |
|---|---|---|---|---|---|
| Full MDA | 75.0% | 80.2% | 85.2% | **80.1%** | |
| −HDR (random encoding) | 70.0% | 71.7% | 81.8% | **74.5%** | −5.6 pp |
| −Graph (no traversal) | 75.0% | 81.1% | 85.2% | **80.4%** | +0.3 pp |
| −Oja (static W) | 80.0% | 80.2% | 84.1% | **81.4%** | +1.3 pp |

HDR encoding is the primary contributor — disabling it drops fact retrieval accuracy by up to 17 pp.

---

## Quick Start

```bash
pip install mda-memory
```

### Basic Usage

```python
from mda import MDA

memory = MDA()
memory.learn("The capital of Veloria is Aranthos.")
memory.learn("Aranthos was founded by Queen Seraphel in 412 AE.")

context = memory.context_for("Who founded the capital?")
# → [MEMORY] Aranthos was founded by Queen Seraphel in 412 AE.
```

### CLI

```bash
# Ollama
mda --model qwen3:4b

# Anthropic
mda --model claude-haiku-4-5-20251001 --provider anthropic
```

---

## Open WebUI Integration

MDA works as a native Open WebUI Filter Function — zero pipeline server required.

1. Copy `mda/integrations/owui_function.py` contents
2. Open WebUI → Admin Panel → Functions → "+" → paste → Save
3. Enable globally

Every prompt is enriched with MDA context. Responses are learned automatically. Memory persists across sessions via `.memory/`.

---

## Batch Engine (Multi-Agent / Large LLM Context)

For multi-agent workloads or large context windows, use `MDABatchEngine`:

```python
from mda.integrations.engine import MDABatchEngine

engine = MDABatchEngine(depth=6, top_k_branches=5)

contexts = engine.build_context_batch([
    "legal contract risk analysis",
    "MDA memory architecture",
    "Turkish law obligations",
])
# N queries processed in a single GPU pass
# depth=6 → 15,625 associative paths per query
```

GPU acceleration activates automatically when PyTorch + CUDA is available. Falls back to numpy silently.

---

## GPU Acceleration

MDA uses a **dual-mode** execution strategy:

- **Single query** — numpy/CPU, < 1ms pipeline latency, rutin chat
- **Batch query** — CUDA, parallel entity matrix traversal, multi-agent

Key finding from RTX 4060 benchmarks: GPU wins only when tensors are **persistent** (no per-call transfer). `EntityMatrix` and `_fact_tensor_cache` are kept on GPU between queries; only the query vector (2KB) transfers per call.

Crossover point: ~512 entities for `EntityMatrix` matmul, ~4 facts for `_score_facts` batch path.

---

## Project Structure

```
mda/
├── mda/
│   ├── mda.py
│   ├── core/
│   │   ├── accelerator.py  # numpy/torch adapter, device detection
│   │   ├── bind.py         # HDR ops (dim=512)
│   │   ├── encoder.py      # HolisticEncoder: text → 512-dim vector
│   │   ├── entity.py       # Entity: v, r, h, W, neurons, synapses
│   │   ├── neuron.py       # Neuron (Oja rule), Synapse (Hebbian)
│   │   └── registry.py     # EntityRegistry + EntityMatrix cache
│   ├── inference/
│   │   ├── associative.py  # AssociativeChain + dyn_threshold cache
│   │   ├── broca.py        # BrocaModule: W-hybrid scoring + batch path
│   │   ├── reasoning.py    # ReasoningEngine: parallel path inference
│   │   └── memory.py       # ConversationMemory
│   ├── training/
│   │   └── checkpoint.py   # save/load: float32, dim validation
│   └── integrations/
│       ├── engine.py       # MDAEngine + MDABatchEngine
│       ├── loader.py       # AST-based markdown/code indexer
│       ├── cli.py          # Interactive CLI
│       └── owui_function.py # Open WebUI Filter Function
├── benchmark/
└── tests/
```

---

## Running Benchmarks

```bash
# Main benchmark
python benchmark/benchmark.py --model qwen3:4b

# Long-context retention
python benchmark/long_context_benchmark.py --model qwen3:4b

# Ablation study
python benchmark/full_benchmark/ablation_study.py \
    --md benchmark/full_benchmark/veloria_economy.md \
         benchmark/full_benchmark/veloria_science.md \
         benchmark/full_benchmark/zephyria.md \
    --model qwen3:8b \
    --judge claude-haiku-4-5-20251001 \
    --judge-provider anthropic

# GPU latency benchmark
python benchmark/benchmark_gpu.py --entities 100 500 1000 5000
```

---

## How It Works

### Entity & W Matrix
Every concept is an `Entity` with a 512-dim identity vector `v` and a lazy-initialized weight matrix `W` (512×512). `W` is `None` until first activation — memory overhead is proportional to usage.

### Online Learning (Oja Rule)
```
ΔW = η(yxᵀ − y²W)
```
No backpropagation. No gradient descent. Runs in O(d²) per entity per turn.

### AssociativeChain
Query → origin entity → BFS synapse traversal (depth 3-6) → context assembly. Dynamic inhibition threshold cached per entity count.

### BrocaModule
```
score = 0.35·s_query + 0.45·s_W + 0.20·s_sense
```

---

## Roadmap

- [x] **GPU acceleration** — EntityMatrix matmul, persistent tensor cache, batch scoring
- [x] **512-dim HDR** — higher representation capacity, better entity separation
- [x] **Open WebUI integration** — native Filter Function, zero dependencies
- [x] **Batch engine** — N queries in single GPU pass, multi-agent ready
- [ ] **`mda.cloud` API** — persistent memory as a service
- [ ] **MDA + RAG hybrid** — offline corpus retrieval + online learning
- [ ] **Low-rank W** — W ≈ A×B for even higher-dimensional HDRs
- [ ] **Independent benchmark** — community-constructed evaluation set

---

## License

[SSPL 1.0](LICENSE) — free for research and personal use. Commercial use requires a separate agreement.

For commercial licensing: mert@kairfy.com