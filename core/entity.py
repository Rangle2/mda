import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from core.bind import DIM, random_vector, zero_vector, normalize, bind, bind_many, cosine
from core.neuron import Neuron, Synapse

MAX_SENSES = 8

@dataclass
class Sense:
    v:             np.ndarray                          # identity vector for this sense
    label:         str        = ""                     # e.g. "finance", "river", "web_internet"
    strength:      float      = 1.0                    # dominance level (increases/decreases with learning)
    use_count:     int        = 0                      # how many times this sense was selected
    context_hints: list       = field(default_factory=list)  # e.g. ["money", "loan"]


ETA_FAST  = 0.05
ETA_SLOW  = 0.001
ETA_META  = 0.010
GAMMA_BASE = 0.95

ROLE_WEIGHTS = {
    "agent":    1.0,
    "patient":  0.3,
    "force":    0.6,
    "location": 0.1,
}


def _det_seed(surface: str) -> int:
    key = surface.lower().encode("utf-8")[:8].ljust(8, b"\0")
    return int.from_bytes(key, "little") % (2 ** 31)


@dataclass
class Entity:
    id:      str
    surface: str

    v:       np.ndarray = field(default_factory=lambda: zero_vector())
    r:       np.ndarray = field(default_factory=lambda: zero_vector())
    h:       np.ndarray = field(default_factory=lambda: zero_vector())
    a:       np.ndarray = field(default_factory=lambda: zero_vector())
    W:       np.ndarray = field(default=None)

    category:  str   = "unknown"
    use_count: int   = 0
    beta:      float = 0.3
    epsilon:   float = 0.3
    last_activated: float = field(default_factory=time.time)

    rho:     dict = field(default_factory=dict)
    gamma_t: np.ndarray = field(default_factory=lambda: zero_vector())
    mu:      np.ndarray = field(default_factory=lambda: zero_vector())

    dim:              int   = DIM
    min_neurons:      int   = 8
    max_neurons:      int   = 64
    growth_threshold: float = 0.75

    neurons:  list  = field(default_factory=list)
    synapses: dict  = field(default_factory=dict)
    senses:   list  = field(default_factory=list)   # Sense list (multi-sense)

    def __post_init__(self):
        seed = _det_seed(self.surface)
        if np.allclose(self.v, 0):
            self.v = random_vector(DIM, seed=seed)
        if np.allclose(self.r, 0):
            self.r = random_vector(DIM, seed=seed + 1)
        if not self.neurons:
            self.neurons = [
                Neuron(dim=self.dim, seed=seed + i + 2)
                for i in range(self.min_neurons)
            ]

    def _ensure_W(self) -> None:
        if self.W is None:
            self.W = np.zeros((self.dim, self.dim))

    def ensemble_activation(self, input_vec: np.ndarray) -> np.ndarray:
        self._ensure_W()
        alive = [n for n in self.neurons if n.is_alive()]
        if not alive:
            raw  = np.tanh(self.W @ self.v)
            norm = np.linalg.norm(raw)
            return raw / (norm + 1e-8) if norm > 1e-6 else zero_vector()

        weighted_sum = np.zeros(self.dim)
        total_weight = 0.0
        for neuron in alive:
            act = neuron.fire(input_vec)
            w   = abs(act)
            if w > 1e-8:
                weighted_sum += w * neuron.weight
                total_weight += w
            neuron.hebbian_update(input_vec)

        if total_weight < 1e-8:
            raw  = np.tanh(self.W @ self.v)
            norm = np.linalg.norm(raw)
            return raw / (norm + 1e-8) if norm > 1e-6 else zero_vector()

        result = weighted_sum / total_weight
        norm   = np.linalg.norm(result)
        return result / (norm + 1e-8) if norm > 1e-6 else zero_vector()

    def predict(self) -> np.ndarray:
        return self.ensemble_activation(self.v)

    def grow(self, encoder) -> None:
        if self.use_count == 0 or self.use_count % 50 != 0:
            return
        if self.beta <= self.growth_threshold:
            return
        if len(self.neurons) >= self.max_neurons:
            return
        alive = [n for n in self.neurons if n.is_alive()]
        if not alive:
            return
        mean_weight = np.mean([n.weight for n in alive], axis=0)
        rng         = np.random.default_rng(self.use_count + len(self.neurons))
        noise       = rng.normal(0, 0.05, self.dim)
        new_neuron  = Neuron(dim=self.dim, seed=self.use_count + len(self.neurons))
        new_neuron.weight = normalize(mean_weight + noise)
        self.neurons.append(new_neuron)

    def prune(self) -> None:
        alive = [n for n in self.neurons if n.is_alive()]
        if len(alive) >= self.min_neurons:
            self.neurons = alive
        else:
            sorted_neurons = sorted(self.neurons, key=lambda n: -n.strength)
            self.neurons   = sorted_neurons[:self.min_neurons]

    def dominant_sense(self, context_vec: np.ndarray = None) -> np.ndarray:
        """Return the most suitable sense vector for the given context."""
        if not self.senses:
            return self.v

        if context_vec is None:
            return max(self.senses, key=lambda s: s.strength).v

        best_sense = None
        best_score = -1.0
        for sense in self.senses:
            score    = float(np.dot(normalize(sense.v), normalize(context_vec)))
            combined = 0.6 * score + 0.4 * sense.strength
            if combined > best_score:
                best_score = combined
                best_sense = sense
        if best_sense:
            self.update_sense_strengths(best_sense.v)
            return best_sense.v
        return self.v

    def add_sense(self, label: str, context_vec: np.ndarray,
                  context_hints: list = None) -> "Sense":
        """Add a new sense; if a very similar sense exists (cosine > 0.80), strengthen it instead.

        Sense vector = the direction of the context stripped of the entity's own identity.
        This captures the unique content of the context, not the structural similarity of sentences.
        """
        context_v = normalize(context_vec)
        raw       = context_v - 0.5 * self.v
        new_v     = normalize(raw) if np.linalg.norm(raw) > 1e-6 else context_v

        for sense in self.senses:
            if cosine(sense.v, new_v) > 0.72:
                sense.strength = min(sense.strength + 0.05, 1.0)
                sense.use_count += 1
                if context_hints:
                    sense.context_hints.extend(
                        h for h in context_hints if h not in sense.context_hints
                    )
                return sense
        new_sense = Sense(
            v=new_v,
            label=label,
            strength=0.3,
            use_count=1,
            context_hints=context_hints or [],
        )
        self.senses.append(new_sense)
        if len(self.senses) > MAX_SENSES:
            self.senses.remove(min(self.senses, key=lambda s: s.strength))
        return new_sense

    def update_sense_strengths(self, winning_sense_v: np.ndarray) -> None:
        """Winning sense is strengthened, others are weakened (lateral inhibition)."""
        for sense in self.senses:
            if cosine(sense.v, winning_sense_v) > 0.8:
                sense.strength = min(sense.strength + 0.02, 1.0)
            else:
                sense.strength = max(sense.strength - 0.01, 0.05)

    def add_synapse(self, other_entity: "Entity", bind_fn) -> None:
        key = other_entity.id
        if key not in self.synapses:
            syn_vec          = bind_fn(self.v, other_entity.v)
            self.synapses[key] = Synapse(
                source_id=self.id,
                target_id=other_entity.id,
                vector=syn_vec,
            )

    def update_synapses(self, other_entities: list, input_vec: np.ndarray) -> None:
        for other in other_entities:
            if other.id == self.id:
                continue
            key = other.id
            if key not in self.synapses:
                self.synapses[key] = Synapse(
                    source_id=self.id,
                    target_id=other.id,
                    vector=bind(self.v, other.v),
                )
            syn = self.synapses[key]
            # Co-occurrence: both entities appeared in same context → fire together, wire together
            syn.hebbian_update(1.0, 1.0, lr=0.02)
            syn.decay()
        self.synapses = {
            k: s for k, s in self.synapses.items()
            if s.is_alive() and not s.prune()
        }

    def neuron_summary(self) -> dict:
        alive = [n for n in self.neurons if n.is_alive()]
        return {
            "total":           len(self.neurons),
            "alive":           len(alive),
            "mean_strength":   float(np.mean([n.strength for n in self.neurons])) if self.neurons else 0.0,
            "mean_activation": float(np.mean([abs(n.activation) for n in alive])) if alive else 0.0,
        }

    def update_W(self, target: np.ndarray, role: str = "agent") -> float:
        self._ensure_W()
        lw      = ROLE_WEIGHTS.get(role, 0.5)
        lr      = ETA_FAST * min(1.0 + self.use_count * 0.05, 3.0)
        l2      = 0.01
        pred    = np.tanh(self.W @ self.v)
        error   = pred - target
        loss    = float(np.mean(error ** 2))
        dtanh   = 1.0 - pred ** 2
        grad    = np.outer(error * dtanh, self.v) + l2 * self.W
        self.W -= lw * lr * grad
        mx      = np.max(np.abs(self.W))
        if mx > 10.0:
            self.W /= mx
        return loss

    def update_memory(self, S: np.ndarray) -> None:
        gamma   = min(GAMMA_BASE + self.use_count * 0.001, 0.999)
        S_norm  = S / (np.linalg.norm(S) + 1e-8)
        self.h  = gamma * self.h + (1 - gamma) * S_norm
        h_norm  = np.linalg.norm(self.h)
        if h_norm > 1e-6:
            self.h /= h_norm

    def update_identity(self, delta: np.ndarray) -> None:
        self.v = normalize(self.v + ETA_SLOW * self.beta * delta)

    def update_meta(self, delta: np.ndarray) -> None:
        self.mu = bind(self.v, delta)
        correction = bind(self.mu, normalize(-self.a + 1e-8))
        self.v = normalize(self.v + ETA_META * correction)

    def update_beta(self, strength: float) -> None:
        gain      = 0.02 * (1.0 + strength)
        self.beta = float(np.clip(self.beta + gain, 0.0, 1.0))

    def update_relation(self, other_id: str, v_other: np.ndarray, strength: float) -> None:
        if other_id not in self.rho:
            self.rho[other_id] = zero_vector()
        self.rho[other_id] = normalize(
            self.rho[other_id] + strength * bind(self.v, v_other)
        )

    def update_transition(self, a_prev: np.ndarray, kappa: np.ndarray, a_next: np.ndarray) -> None:
        self.gamma_t = bind_many(a_prev, kappa, a_next)

    def contrastive_update(self, pos: np.ndarray, negs: list[np.ndarray]) -> None:
        if not negs:
            return
        self._ensure_W()
        pred     = np.tanh(self.W @ self.v)
        sim_pos  = float(np.dot(pred, pos))
        sim_negs = np.array([float(np.dot(pred, n)) for n in negs])
        T        = 0.1
        exp_pos  = np.exp(sim_pos / T)
        exp_negs = np.exp(sim_negs / T)
        denom    = exp_pos + np.sum(exp_negs)
        d_pred   = -(1 - exp_pos / denom) * pos / T
        for i, n in enumerate(negs):
            d_pred += (exp_negs[i] / denom) * n / T
        dtanh    = 1.0 - pred ** 2
        grad     = np.outer(d_pred * dtanh, self.v)
        self.W  -= ETA_FAST * 0.3 * grad
        mx       = np.max(np.abs(self.W))
        if mx > 10.0:
            self.W /= mx

    def decay(self) -> None:
        import math
        delta_days = (time.time() - self.last_activated) / 86400
        temporal_factor = math.exp(-0.01 * delta_days)
        self.beta = float(np.clip(self.beta * 0.995 * temporal_factor, 0.05, 1.0))
        self.last_activated = time.time()

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "surface":   self.surface,
            "category":  self.category,
            "use_count": self.use_count,
            "beta":      round(self.beta, 4),
        }
