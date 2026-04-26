from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections import deque

import numpy as np
from mda.core.bind import cosine, normalize


MAX_TURNS    = 20
DECAY_RATE   = 0.85
MIN_SALIENCE = 0.15


@dataclass
class Turn:
    role:      str
    text:      str
    vector:    np.ndarray
    entities:  list[str]
    timestamp: float = field(default_factory=time.time)
    salience:  float = 1.0


class ConversationMemory:
    """
    Context tracking: stores each turn's vector and entities.
    Returns the most relevant past turns at query time,
    prunes stale context, and detects repeated questions.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self._turns:   deque[Turn] = deque(maxlen=max_turns)
        self._ctx_vec: np.ndarray | None = None

    def add(self, role: str, text: str, vector: np.ndarray,
            entities: list[str] | None = None) -> None:
        turn = Turn(
            role=role,
            text=text,
            vector=normalize(vector.copy()),
            entities=entities or [],
        )
        self._turns.append(turn)
        self._update_ctx(turn.vector)

    def _update_ctx(self, v: np.ndarray) -> None:
        if self._ctx_vec is None:
            self._ctx_vec = v.copy()
        else:
            self._ctx_vec = normalize(DECAY_RATE * self._ctx_vec + (1 - DECAY_RATE) * v)

    def context_vector(self) -> np.ndarray | None:
        return self._ctx_vec

    def relevant_turns(self, query_vec: np.ndarray, top_k: int = 3) -> list[Turn]:
        if not self._turns:
            return []
        qv = normalize(query_vec)
        scored = []
        age_factor = 1.0
        for turn in reversed(self._turns):
            sim = float(cosine(qv, turn.vector)) * turn.salience * age_factor
            scored.append((sim, turn))
            age_factor *= 0.9
        scored.sort(key=lambda x: -x[0])
        return [t for s, t in scored[:top_k] if s > MIN_SALIENCE]

    def active_entities(self, top_k: int = 5) -> list[str]:
        counts: dict[str, float] = {}
        weight = 1.0
        for turn in reversed(self._turns):
            for ent in turn.entities:
                counts[ent] = counts.get(ent, 0.0) + weight
            weight *= DECAY_RATE
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        return [e for e, _ in ranked[:top_k]]

    def is_repeat(self, query_vec: np.ndarray, threshold: float = 0.93) -> bool:
        qv = normalize(query_vec)
        for turn in self._turns:
            if turn.role == "user" and cosine(qv, turn.vector) >= threshold:
                return True
        return False

    def decay_salience(self) -> None:
        for turn in self._turns:
            turn.salience = max(turn.salience * DECAY_RATE, MIN_SALIENCE)

    def summary(self, max_chars: int = 400) -> str:
        if not self._turns:
            return ""
        lines = []
        for turn in list(self._turns)[-6:]:
            prefix = "User" if turn.role == "user" else "MDA"
            lines.append(f"{prefix}: {turn.text[:120]}")
        text = "\n".join(lines)
        return text[:max_chars]

    def recent_entities(self, n: int = 3) -> list[str]:
        seen: list[str] = []
        for turn in reversed(self._turns):
            for ent in turn.entities:
                if ent not in seen:
                    seen.append(ent)
                if len(seen) >= n:
                    return seen
        return seen

    def clear(self) -> None:
        self._turns.clear()
        self._ctx_vec = None

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={len(self._turns)})"
