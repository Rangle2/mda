"""Tests for core/entity.py — Entity and Sense."""
import time
import numpy as np
import pytest
from core.bind import DIM, random_vector, normalize, cosine
from core.entity import Entity, Sense


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entity(surface="Widget", eid="e_000001"):
    return Entity(id=eid, surface=surface)


# ---------------------------------------------------------------------------
# Entity construction
# ---------------------------------------------------------------------------

class TestEntityConstruction:
    def test_identity_vector_not_zero(self):
        e = make_entity()
        assert not np.allclose(e.v, 0)

    def test_identity_vector_unit_norm(self):
        e = make_entity()
        assert abs(np.linalg.norm(e.v) - 1.0) < 1e-5

    def test_deterministic_vector_by_surface(self):
        e1 = make_entity("Alpha", "e_0")
        e2 = make_entity("Alpha", "e_1")
        np.testing.assert_array_equal(e1.v, e2.v)

    def test_different_surfaces_differ(self):
        e1 = make_entity("Alpha")
        e2 = make_entity("Beta")
        assert not np.allclose(e1.v, e2.v)

    def test_default_neuron_count(self):
        e = make_entity()
        assert len(e.neurons) == e.min_neurons

    def test_default_category(self):
        e = make_entity()
        assert e.category == "unknown"

    def test_last_activated_set_on_creation(self):
        before = time.time()
        e = make_entity()
        assert e.last_activated >= before


# ---------------------------------------------------------------------------
# update_beta / decay
# ---------------------------------------------------------------------------

class TestBetaAndDecay:
    def test_update_beta_increases(self):
        e = make_entity()
        before = e.beta
        e.update_beta(0.5)
        assert e.beta > before

    def test_update_beta_capped_at_one(self):
        e = make_entity()
        e.beta = 0.99
        e.update_beta(1.0)
        assert e.beta <= 1.0

    def test_decay_reduces_beta(self):
        e = make_entity()
        e.beta = 0.8
        e.decay()
        assert e.beta < 0.8

    def test_decay_beta_clamp_floor(self):
        e = make_entity()
        e.beta = 0.001
        e.decay()
        assert e.beta >= 0.05

    def test_decay_updates_last_activated(self):
        e = make_entity()
        old_ts = e.last_activated - 10
        e.last_activated = old_ts
        e.decay()
        assert e.last_activated > old_ts

    def test_decay_temporal_factor_older_entity(self):
        e1 = make_entity("A")
        e2 = make_entity("B")
        e1.beta = 0.8
        e2.beta = 0.8
        # e1 was activated recently, e2 was activated 10 days ago
        e1.last_activated = time.time()
        e2.last_activated = time.time() - 10 * 86400
        e1.decay()
        e2.decay()
        # Older entity should decay more
        assert e2.beta < e1.beta


# ---------------------------------------------------------------------------
# update_W / update_memory
# ---------------------------------------------------------------------------

class TestWeightAndMemory:
    def test_update_W_creates_matrix(self):
        e = make_entity()
        target = random_vector(DIM, seed=1)
        assert e.W is None
        e.update_W(target)
        assert e.W is not None
        assert e.W.shape == (DIM, DIM)

    def test_update_W_returns_loss(self):
        e = make_entity()
        target = random_vector(DIM, seed=2)
        loss = e.update_W(target)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_update_W_reduces_loss_over_iterations(self):
        e = make_entity()
        e.use_count = 10
        target = normalize(random_vector(DIM, seed=3))
        losses = [e.update_W(target) for _ in range(20)]
        assert losses[-1] <= losses[0] + 0.5  # should generally converge

    def test_update_memory_changes_h(self):
        e = make_entity()
        original_h = e.h.copy()
        v = random_vector(DIM, seed=4)
        e.update_memory(v)
        assert not np.allclose(e.h, original_h)

    def test_update_memory_h_stays_normalized(self):
        e = make_entity()
        for seed in range(5):
            e.update_memory(random_vector(DIM, seed=seed))
        norm = np.linalg.norm(e.h)
        assert norm < 1.0 + 1e-5


# ---------------------------------------------------------------------------
# add_sense / dominant_sense
# ---------------------------------------------------------------------------

class TestSenses:
    def test_add_sense_creates_sense(self):
        e = make_entity()
        ctx = random_vector(DIM, seed=10)
        s = e.add_sense("test", ctx)
        assert len(e.senses) == 1
        assert s.label == "test"

    def test_add_similar_sense_strengthens_existing(self):
        e = make_entity()
        ctx = random_vector(DIM, seed=11)
        s1 = e.add_sense("a", ctx)
        initial_strength = s1.strength
        s2 = e.add_sense("b", ctx)  # same direction → should merge
        assert s2 is s1
        assert s1.strength > initial_strength

    def test_add_dissimilar_sense_creates_new(self):
        e = make_entity()
        ctx1 = random_vector(DIM, seed=12)
        ctx2 = -ctx1  # opposite direction
        e.add_sense("a", ctx1)
        e.add_sense("b", ctx2)
        assert len(e.senses) == 2

    def test_dominant_sense_no_senses_returns_v(self):
        e = make_entity()
        dom = e.dominant_sense()
        np.testing.assert_array_equal(dom, e.v)

    def test_dominant_sense_with_context(self):
        e = make_entity()
        ctx = random_vector(DIM, seed=13)
        e.add_sense("ctx", ctx)
        result = e.dominant_sense(ctx)
        assert result.shape == (DIM,)


# ---------------------------------------------------------------------------
# Synapses
# ---------------------------------------------------------------------------

class TestSynapses:
    def test_add_synapse_creates_entry(self):
        from core.bind import bind
        e1 = make_entity("A", "e_0")
        e2 = make_entity("B", "e_1")
        e1.add_synapse(e2, bind)
        assert e2.id in e1.synapses

    def test_add_synapse_idempotent(self):
        from core.bind import bind
        e1 = make_entity("A", "e_0")
        e2 = make_entity("B", "e_1")
        e1.add_synapse(e2, bind)
        e1.add_synapse(e2, bind)
        assert len(e1.synapses) == 1

    def test_update_synapses_adds_connection(self):
        e1 = make_entity("A", "e_0")
        e2 = make_entity("B", "e_1")
        v = random_vector(DIM, seed=20)
        e1.update_synapses([e2], v)
        assert e2.id in e1.synapses


# ---------------------------------------------------------------------------
# grow / prune / ensemble_activation
# ---------------------------------------------------------------------------

class TestGrowPruneActivation:
    def test_grow_does_not_add_unless_criteria_met(self):
        e = make_entity()
        initial = len(e.neurons)
        e.grow(None)  # use_count == 0, should not grow
        assert len(e.neurons) == initial

    def test_grow_adds_neuron_when_criteria_met(self):
        e = make_entity()
        e.use_count = 50
        e.beta = 0.9  # above growth threshold
        enc_stub = None
        e.grow(enc_stub)
        assert len(e.neurons) > e.min_neurons

    def test_prune_keeps_min_neurons(self):
        e = make_entity()
        for n in e.neurons:
            n.strength = 0.0  # kill all
        e.prune()
        assert len(e.neurons) == e.min_neurons

    def test_ensemble_activation_returns_vector(self):
        e = make_entity()
        v = random_vector(DIM, seed=30)
        result = e.ensemble_activation(v)
        assert result.shape == (DIM,)
        assert np.all(np.isfinite(result))

    def test_predict_returns_vector(self):
        e = make_entity()
        result = e.predict()
        assert result.shape == (DIM,)

    def test_neuron_summary_keys(self):
        e = make_entity()
        summary = e.neuron_summary()
        assert "total" in summary
        assert "alive" in summary
        assert "mean_strength" in summary


# ---------------------------------------------------------------------------
# contrastive_update
# ---------------------------------------------------------------------------

class TestContrastiveUpdate:
    def test_no_negs_is_noop(self):
        e = make_entity()
        target = random_vector(DIM, seed=50)
        e.update_W(target)
        W_before = e.W.copy()
        e.contrastive_update(target, [])
        np.testing.assert_array_equal(e.W, W_before)

    def test_with_negs_modifies_W(self):
        e = make_entity()
        pos = random_vector(DIM, seed=51)
        negs = [random_vector(DIM, seed=i + 52) for i in range(3)]
        e.update_W(pos)
        W_before = e.W.copy()
        e.contrastive_update(pos, negs)
        assert not np.allclose(e.W, W_before)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_keys(self):
        e = make_entity("Foo")
        d = e.to_dict()
        assert "id" in d
        assert "surface" in d
        assert "category" in d
        assert "use_count" in d
        assert "beta" in d

    def test_to_dict_surface_matches(self):
        e = make_entity("Bar")
        assert e.to_dict()["surface"] == "Bar"
