"""Tests for core/neuron.py — Neuron and Synapse."""
import time
import numpy as np
import pytest
from mda.core.bind import DIM, random_vector, normalize
from mda.core.neuron import Neuron, Synapse


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------

class TestNeuron:
    def test_initial_strength(self):
        n = Neuron(dim=DIM, seed=0)
        assert n.strength == 0.5

    def test_weight_is_unit_norm(self):
        n = Neuron(dim=DIM, seed=1)
        assert abs(np.linalg.norm(n.weight) - 1.0) < 1e-6

    def test_fire_returns_float(self):
        n = Neuron(dim=DIM, seed=2)
        v = random_vector(DIM, seed=2)
        result = n.fire(v)
        assert isinstance(result, float)

    def test_fire_increments_age(self):
        n = Neuron(dim=DIM, seed=3)
        v = random_vector(DIM, seed=3)
        n.fire(v)
        assert n.age == 1
        n.fire(v)
        assert n.age == 2

    def test_fire_result_bounded(self):
        n = Neuron(dim=DIM, seed=4)
        for seed in range(10):
            v = random_vector(DIM, seed=seed)
            r = n.fire(v)
            assert -1.0 <= r <= 1.0

    def test_fire_updates_activation_when_above_threshold(self):
        n = Neuron(dim=DIM, seed=5)
        # Force a high-activation input by aligning weight and input
        v = n.weight.copy()
        val = n.fire(v)
        if abs(val) >= n.threshold:
            assert n.activation == pytest.approx(val)

    def test_hebbian_update_keeps_unit_norm(self):
        n = Neuron(dim=DIM, seed=6)
        v = random_vector(DIM, seed=6)
        n.hebbian_update(v)
        assert abs(np.linalg.norm(n.weight) - 1.0) < 1e-6

    def test_decay_reduces_strength(self):
        n = Neuron(dim=DIM, seed=7)
        before = n.strength
        n.decay(0.9)
        assert n.strength < before

    def test_strengthen_increases_strength(self):
        n = Neuron(dim=DIM, seed=8)
        n.strength = 0.5
        n.strengthen(0.1)
        assert n.strength == pytest.approx(0.6)

    def test_strengthen_caps_at_one(self):
        n = Neuron(dim=DIM, seed=9)
        n.strength = 0.99
        n.strengthen(0.5)
        assert n.strength == pytest.approx(1.0)

    def test_is_alive_true_by_default(self):
        n = Neuron(dim=DIM, seed=10)
        assert n.is_alive()

    def test_is_alive_false_after_decay_to_zero(self):
        n = Neuron(dim=DIM, seed=11)
        n.strength = 0.04
        assert not n.is_alive()

    def test_deterministic_weight_with_seed(self):
        n1 = Neuron(dim=DIM, seed=42)
        n2 = Neuron(dim=DIM, seed=42)
        np.testing.assert_array_equal(n1.weight, n2.weight)

    def test_different_seeds_different_weights(self):
        n1 = Neuron(dim=DIM, seed=1)
        n2 = Neuron(dim=DIM, seed=2)
        assert not np.allclose(n1.weight, n2.weight)


# ---------------------------------------------------------------------------
# Synapse
# ---------------------------------------------------------------------------

class TestSynapse:
    @pytest.fixture
    def synapse(self):
        vec = random_vector(DIM, seed=0)
        return Synapse(source_id="src", target_id="tgt", vector=vec)

    def test_initial_strength(self, synapse):
        assert synapse.strength == 0.1

    def test_fire_returns_ndarray(self, synapse):
        v = random_vector(DIM, seed=1)
        result = synapse.fire(v)
        assert isinstance(result, np.ndarray)
        assert result.shape == (DIM,)

    def test_fire_increments_activation_count(self, synapse):
        v = random_vector(DIM, seed=2)
        synapse.fire(v)
        assert synapse.activation_count == 1
        synapse.fire(v)
        assert synapse.activation_count == 2

    def test_fire_updates_last_activated(self, synapse):
        before = synapse.last_activated
        time.sleep(0.01)
        v = random_vector(DIM, seed=3)
        synapse.fire(v)
        assert synapse.last_activated >= before

    def test_hebbian_update_clamps_strength(self, synapse):
        synapse.hebbian_update(1.0, 1.0, lr=100.0)
        assert synapse.strength <= 1.0
        synapse2 = Synapse("s", "t", random_vector(DIM, seed=0))
        synapse2.hebbian_update(-1.0, 1.0, lr=100.0)
        assert synapse2.strength >= 0.0

    def test_decay_reduces_strength(self, synapse):
        before = synapse.strength
        synapse.decay(0.5)
        assert synapse.strength < before

    def test_is_alive_true_initially(self, synapse):
        assert synapse.is_alive()

    def test_is_alive_false_when_weak(self, synapse):
        synapse.strength = 0.005
        assert not synapse.is_alive()

    def test_prune_true_when_old_and_weak(self, synapse):
        synapse._age = 1001
        synapse.strength = 0.01
        assert synapse.prune()

    def test_prune_false_when_young(self, synapse):
        synapse._age = 10
        synapse.strength = 0.01
        assert not synapse.prune()

    def test_decayed_strength_decreases_over_time(self, synapse):
        synapse.last_activated = time.time() - 86400  # 1 day ago
        ds = synapse.decayed_strength()
        assert ds < synapse.strength

    def test_apply_decay_bakes_in_decay(self, synapse):
        synapse.last_activated = time.time() - 86400
        original = synapse.strength
        synapse.apply_decay()
        assert synapse.strength < original

    def test_vector_is_copy_not_reference(self):
        vec = random_vector(DIM, seed=0)
        syn = Synapse("s", "t", vec)
        vec[:] = 0
        assert not np.allclose(syn.vector, 0)
