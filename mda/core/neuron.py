import math
import time
import numpy as np
from mda.core.bind import bind, normalize, DIM


class Neuron:
    def __init__(self, dim: int = DIM, seed: int = None):
        self.id: str = f"n_{abs(hash((dim, seed, id(self)))):x}"
        rng = np.random.default_rng(seed)
        self.weight: np.ndarray = normalize(rng.normal(0, 1, dim))
        self.activation: float = 0.0
        self.threshold: float = 0.3
        self.age: int = 0
        self.strength: float = 0.5

    def fire(self, input_vec: np.ndarray) -> float:
        val = float(np.tanh(np.dot(self.weight, input_vec)))
        if abs(val) >= self.threshold:
            self.activation = val
        self.age += 1
        return val

    def hebbian_update(self, input_vec: np.ndarray, lr: float = 0.05) -> None:
        act = float(np.dot(self.weight, input_vec))
        delta = lr * act * (input_vec - act * self.weight)
        self.weight = normalize(self.weight + delta)
        if abs(act) >= self.threshold:
            self.strengthen(0.01)
        else:
            self.decay(0.999)

    def decay(self, rate: float = 0.995) -> None:
        self.strength *= rate

    def strengthen(self, amount: float = 0.02) -> None:
        self.strength = min(self.strength + amount, 1.0)

    def is_alive(self) -> bool:
        return self.strength > 0.05


class Synapse:
    def __init__(self, source_id: str, target_id: str, vector: np.ndarray):
        self.source_id: str = source_id
        self.target_id: str = target_id
        self.strength: float = 0.1
        self.vector: np.ndarray = vector.copy()
        self.activation_count: int = 0
        self._age: int = 0
        self.last_activated: float = time.time()

    def fire(self, input_vec: np.ndarray) -> np.ndarray:
        self.activation_count += 1
        self._age += 1
        self.last_activated = time.time()
        return self.strength * bind(self.vector, input_vec)

    def decayed_strength(self, now: float | None = None) -> float:
        """Effective strength after temporal decay: strength * exp(-0.01 * delta_days)."""
        if now is None:
            now = time.time()
        delta_days = max(0.0, (now - self.last_activated) / 86400.0)
        return self.strength * math.exp(-0.01 * delta_days)

    def apply_decay(self, now: float | None = None) -> None:
        """Bake in the accumulated decay permanently and reset last_activated."""
        if now is None:
            now = time.time()
        self.strength = self.decayed_strength(now)
        self.last_activated = now

    def hebbian_update(self, source_act: float, target_act: float, lr: float = 0.005) -> None:
        self.strength = float(np.clip(self.strength + lr * source_act * target_act, 0.0, 1.0))

    def decay(self, rate: float = 0.999) -> None:
        self.strength *= rate

    def is_alive(self) -> bool:
        return self.strength > 0.01

    def prune(self) -> bool:
        return self._age > 1000 and self.strength < 0.02
