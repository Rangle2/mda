from __future__ import annotations

import os as _os

_DATA_DIR = _os.path.dirname(_os.path.abspath(__file__))


def _load_stopwords(filename: str) -> frozenset:
    path = _os.path.join(_DATA_DIR, filename)
    words: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line.lower())
    return frozenset(words)


# stopwords.txt        — general lookup stopwords (all function/question words)
# stopwords_content.txt — words excluded from entity bootstrap during passive learning
LOOKUP_STOPWORDS  = _load_stopwords("stopwords.txt")
CONTENT_STOPWORDS = _load_stopwords("stopwords_content.txt")
