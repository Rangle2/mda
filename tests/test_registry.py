"""Tests for core/registry.py — EntityRegistry."""
import numpy as np
import pytest
from core.bind import DIM, random_vector
from core.registry import EntityRegistry, CATEGORY_VECTORS


class TestGetOrCreate:
    def test_creates_new_entity(self, registry):
        e = registry.get_or_create("Python", "software")
        assert e is not None
        assert e.surface == "Python"

    def test_returns_same_entity_on_repeat(self, registry):
        e1 = registry.get_or_create("Python", "software")
        e2 = registry.get_or_create("Python", "software")
        assert e1.id == e2.id

    def test_repeat_increments_use_count(self, registry):
        registry.get_or_create("X")
        e = registry.get_or_create("X")
        assert e.use_count >= 1

    def test_case_insensitive_dedup(self, registry):
        e1 = registry.get_or_create("Python")
        e2 = registry.get_or_create("python")
        assert e1.id == e2.id

    def test_category_vector_applied_for_known_category(self, registry):
        e = registry.get_or_create("GPT", "software")
        assert not np.allclose(e.v, 0)
        # Should be close to the software category vector (noise std=0.05 on 256-dim)
        from core.bind import cosine
        sim = cosine(e.v, CATEGORY_VECTORS["software"])
        assert sim > 0.7

    def test_unknown_category_still_creates_entity(self, registry):
        e = registry.get_or_create("Mystery", "unknown")
        assert e is not None


class TestGetters:
    def test_get_returns_entity(self, registry):
        registry.get_or_create("Alpha")
        e = registry.get("Alpha")
        assert e is not None
        assert e.surface == "Alpha"

    def test_get_case_insensitive(self, registry):
        registry.get_or_create("Alpha")
        assert registry.get("alpha") is not None
        assert registry.get("ALPHA") is not None

    def test_get_missing_returns_none(self, registry):
        assert registry.get("DoesNotExist") is None

    def test_get_by_id(self, registry):
        e = registry.get_or_create("Beta")
        found = registry.get_by_id(e.id)
        assert found is e

    def test_get_by_id_missing_returns_none(self, registry):
        assert registry.get_by_id("e_999999") is None


class TestCountAndAll:
    def test_count_empty(self):
        r = EntityRegistry()
        assert r.count() == 0

    def test_count_increments(self, registry):
        registry.get_or_create("A")
        registry.get_or_create("B")
        assert registry.count() == 2

    def test_count_dedup_not_double_counted(self, registry):
        registry.get_or_create("A")
        registry.get_or_create("A")
        assert registry.count() == 1

    def test_all_returns_list(self, registry):
        registry.get_or_create("A")
        registry.get_or_create("B")
        result = registry.all()
        assert isinstance(result, list)
        assert len(result) == 2


class TestInferCategory:
    def test_returns_string(self, registry):
        v = random_vector(DIM, seed=0)
        cat = registry.infer_category(v)
        assert isinstance(cat, str)

    def test_known_category_vector_infers_correctly(self, registry):
        v = CATEGORY_VECTORS["software"].copy()
        cat = registry.infer_category(v)
        assert cat == "software"


class TestSummary:
    def test_summary_contains_total(self, registry):
        registry.get_or_create("A", "concept")
        s = registry.summary()
        assert "concept" in s

    def test_summary_empty_registry(self):
        r = EntityRegistry()
        s = r.summary()
        assert "0" in s


class TestPrune:
    def test_prune_removes_low_use_entities(self, registry):
        registry.get_or_create("Weak", "unknown")  # use_count starts at 0, no synapses
        # min_synapse_strength must be > 0 so that 0.0 < threshold → removal condition met
        removed = registry.prune(min_use_count=1, min_synapse_strength=0.01)
        assert removed >= 1
        assert registry.get("Weak") is None

    def test_prune_keeps_high_use_entities(self, registry):
        e = registry.get_or_create("Strong", "unknown")
        e.use_count = 10
        registry.prune(min_use_count=5, min_synapse_strength=0.0)
        assert registry.get("Strong") is not None

    def test_remove_deletes_entity(self, registry):
        e = registry.get_or_create("Gone")
        registry.remove(e)
        assert registry.get("Gone") is None


class TestUpdateSynapsesAll:
    def test_updates_synapses_between_active_entities(self, registry):
        a = registry.get_or_create("A")
        b = registry.get_or_create("B")
        c = registry.get_or_create("C")
        v = random_vector(DIM, seed=0)
        registry.update_synapses_all([a, b, c], v)
        # Each entity should now have synapses to the others
        assert len(a.synapses) > 0

    def test_single_entity_no_synapses(self, registry):
        a = registry.get_or_create("Solo")
        v = random_vector(DIM, seed=0)
        registry.update_synapses_all([a], v)
        assert len(a.synapses) == 0


class TestSaveLoadState:
    def test_round_trip(self, registry):
        e = registry.get_or_create("Rust", "software")
        e.use_count = 7
        e.beta = 0.6
        state = registry.save_state()

        r2 = EntityRegistry()
        r2.load_state(state)
        e2 = r2.get("Rust")
        assert e2 is not None
        assert e2.use_count == 7
        assert e2.beta == pytest.approx(0.6)
