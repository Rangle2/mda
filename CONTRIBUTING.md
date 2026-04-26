# Contributing to MDA

Thank you for your interest in MDA.

## How to Contribute

### Reporting Bugs
Open a GitHub issue with:
- Python version and OS
- Minimal reproduction script
- Expected vs actual behavior

### Submitting PRs
1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/`
4. Open a PR with a clear description

### Priority Areas
These are the highest-value contributions right now:

- **GPU port** — PyTorch tensor ops for HDR encoding and Oja updates
- **Low-rank W** — W ≈ A×B factorization for higher-dimensional HDRs
- **MDA + RAG hybrid** — combining offline retrieval with online learning
- **Real-world benchmarks** — evaluation beyond fictional domains

---

## Code Standards

- No inline comments unless explaining non-obvious math
- Type hints on all public functions
- Tests for any new module in `tests/`

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/
```
