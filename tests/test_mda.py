"""Tests for mda.py — MDA top-level API."""
import numpy as np
import pytest
from mda.core.bind import DIM, normalize


@pytest.fixture
def mda():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mda import MDA
    return MDA(dim=DIM)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_history_initialized(self, mda):
        assert hasattr(mda, "_history")
        assert isinstance(mda._history, list)
        assert mda._history == []

    def test_turn_count_zero(self, mda):
        assert mda._turn_count == 0

    def test_registry_empty(self, mda):
        assert mda.registry.count() == 0

    def test_repr(self, mda):
        r = repr(mda)
        assert "MDA" in r
        assert "dim=" in r


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

class TestEncode:
    def test_returns_ndarray(self, mda):
        v = mda.encode("hello world")
        assert isinstance(v, np.ndarray)

    def test_correct_dim(self, mda):
        v = mda.encode("test")
        assert v.shape == (DIM,)

    def test_normalized(self, mda):
        v = mda.encode("normalized output")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# teach
# ---------------------------------------------------------------------------

class TestTeach:
    def test_creates_entity(self, mda):
        mda.teach("Python", ["Python is a language."])
        assert mda.registry.get("Python") is not None

    def test_stores_facts_in_broca(self, mda):
        mda.teach("Rust", ["Rust is memory-safe."])
        e = mda.registry.get("Rust")
        assert e is not None
        assert len(mda.broca._entity_facts.get(e.id, [])) > 0

    def test_returns_self(self, mda):
        result = mda.teach("Go", ["Go is concurrent."])
        assert result is mda

    def test_category_assigned(self, mda):
        mda.teach("Java", ["Java runs on JVM."], category="software")
        e = mda.registry.get("Java")
        assert e.category == "software"

    def test_multiple_facts(self, mda):
        mda.teach("AI", ["AI stands for Artificial Intelligence.",
                          "AI includes ML and DL."])
        e = mda.registry.get("AI")
        assert e.use_count >= 2


# ---------------------------------------------------------------------------
# relate
# ---------------------------------------------------------------------------

class TestRelate:
    def test_relate_creates_synapses(self, mda):
        mda.teach("Cat", ["Cat is a mammal."])
        mda.teach("Dog", ["Dog is a mammal."])
        mda.relate("Cat", "Dog")
        cat = mda.registry.get("Cat")
        dog = mda.registry.get("Dog")
        assert dog.id in cat.synapses
        assert cat.id in dog.synapses

    def test_relate_missing_entity_is_noop(self, mda):
        mda.teach("Existing", ["It exists."])
        # Should not raise
        mda.relate("Existing", "NonExistent")

    def test_returns_self(self, mda):
        mda.teach("X", ["x"])
        mda.teach("Y", ["y"])
        result = mda.relate("X", "Y")
        assert result is mda


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------

class TestFindSimilar:
    def test_returns_list_of_tuples(self, mda):
        mda.teach("Alpha", ["Alpha is first."])
        mda.teach("Beta",  ["Beta is second."])
        results = mda.find_similar("Alpha", top_k=2)
        assert isinstance(results, list)
        for surface, score in results:
            assert isinstance(surface, str)
            assert isinstance(score, float)

    def test_top_k_respected(self, mda):
        for i in range(5):
            mda.teach(f"Entity{i}", [f"Entity {i} is here."])
        results = mda.find_similar("Entity0", top_k=3)
        assert len(results) <= 3

    def test_scores_descending(self, mda):
        for i in range(4):
            mda.teach(f"E{i}", [f"E{i} is entity {i}."])
        results = mda.find_similar("E0", top_k=4)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_unknown_surface_returns_zero(self, mda):
        assert mda.confidence("UnknownXYZ", "any query") == 0.0

    def test_entity_without_facts_returns_zero(self, mda):
        mda.registry.get_or_create("Factless", "unknown")
        assert mda.confidence("Factless", "any query") == 0.0

    def test_taught_entity_returns_positive(self, mda):
        mda.teach("DataScience", ["Data science uses statistics and ML."])
        score = mda.confidence("DataScience", "data science statistics")
        assert score > 0.0

    def test_result_in_range(self, mda):
        mda.teach("NLP", ["NLP processes natural language."])
        score = mda.confidence("NLP", "natural language")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

class TestLearn:
    def test_returns_self(self, mda):
        result = mda.learn("Neural networks learn from mda.data.")
        assert result is mda

    def test_creates_entities_from_text(self, mda):
        mda.learn("TensorFlow is used for deep learning.")
        # At least some entity should have been created
        assert mda.registry.count() > 0

    def test_memory_records_turn(self, mda):
        before = len(mda._memory)
        mda.learn("Language models understand context.")
        assert len(mda._memory) > before


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------

class TestProcess:
    def test_no_crash_on_empty_text(self, mda):
        mda.process("")  # should not raise

    def test_updates_known_entity(self, mda):
        mda.teach("GPU", ["GPU accelerates training."])
        e = mda.registry.get("GPU")
        before = e.use_count
        mda.process("GPU is fast")
        assert e.use_count >= before


# ---------------------------------------------------------------------------
# experience
# ---------------------------------------------------------------------------

class TestExperience:
    def test_no_crash_on_unknown_words(self, mda):
        mda.experience("zzz aaa bbb ccc")  # no entities → should return early

    def test_memory_updated(self, mda):
        mda.teach("Memory", ["Memory stores information."])
        before = len(mda._memory)
        mda.experience("Memory is important")
        assert len(mda._memory) >= before


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_string(self, mda):
        s = mda.stats()
        assert isinstance(s, str)

    def test_entity_count_in_stats(self, mda):
        mda.teach("Stat", ["Stat is here."])
        s = mda.stats()
        assert "1" in s

    def test_stats_with_history(self, mda):
        mda._history.append(0.25)
        s = mda.stats()
        assert "0.25" in s

    def test_stats_with_dict_history(self, mda):
        mda._history.append({"train": 0.1, "test": 0.2})
        s = mda.stats()
        assert "Train" in s or "0.1" in s
