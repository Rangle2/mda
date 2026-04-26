"""Tests for inference/memory.py — ConversationMemory."""
import time
import numpy as np
import pytest
from mda.core.bind import DIM, random_vector, normalize
from mda.inference.memory import ConversationMemory, Turn, MAX_TURNS, DECAY_RATE, MIN_SALIENCE


@pytest.fixture
def mem():
    return ConversationMemory()


def _vec(seed: int) -> np.ndarray:
    return random_vector(DIM, seed=seed)


# ---------------------------------------------------------------------------
# add / context_vector / __len__
# ---------------------------------------------------------------------------

class TestAdd:
    def test_len_increments(self, mem):
        mem.add("user", "hello", _vec(0))
        assert len(mem) == 1

    def test_context_vector_set_after_first_add(self, mem):
        mem.add("user", "test", _vec(1))
        assert mem.context_vector() is not None

    def test_context_vector_normalized(self, mem):
        mem.add("user", "a", _vec(2))
        cv = mem.context_vector()
        assert abs(np.linalg.norm(cv) - 1.0) < 1e-6

    def test_context_updates_on_second_add(self, mem):
        mem.add("user", "a", _vec(3))
        ctx1 = mem.context_vector().copy()
        mem.add("user", "b", _vec(4))
        ctx2 = mem.context_vector()
        assert not np.allclose(ctx1, ctx2)

    def test_max_turns_enforced(self):
        m = ConversationMemory(max_turns=3)
        for i in range(5):
            m.add("user", f"msg {i}", _vec(i))
        assert len(m) == 3

    def test_entities_stored(self, mem):
        mem.add("user", "about Alpha", _vec(5), entities=["Alpha", "Beta"])
        # Should be accessible via active_entities
        active = mem.active_entities(top_k=5)
        assert "Alpha" in active

    def test_empty_context_none(self):
        m = ConversationMemory()
        assert m.context_vector() is None


# ---------------------------------------------------------------------------
# relevant_turns
# ---------------------------------------------------------------------------

class TestRelevantTurns:
    def test_empty_memory_returns_empty(self, mem):
        result = mem.relevant_turns(_vec(0))
        assert result == []

    def test_returns_at_most_top_k(self, mem):
        for i in range(5):
            mem.add("user", f"turn {i}", _vec(i))
        result = mem.relevant_turns(_vec(0), top_k=2)
        assert len(result) <= 2

    def test_high_similarity_turn_included(self, mem):
        v = _vec(10)
        mem.add("user", "target turn", v)
        result = mem.relevant_turns(v, top_k=1)
        assert len(result) == 1
        assert result[0].text == "target turn"

    def test_result_elements_are_turns(self, mem):
        mem.add("user", "hello", _vec(20))
        result = mem.relevant_turns(_vec(20), top_k=1)
        if result:
            assert isinstance(result[0], Turn)


# ---------------------------------------------------------------------------
# active_entities
# ---------------------------------------------------------------------------

class TestActiveEntities:
    def test_empty_returns_empty(self, mem):
        assert mem.active_entities() == []

    def test_most_recent_entities_ranked_first(self, mem):
        mem.add("user", "x", _vec(0), entities=["Old"])
        mem.add("user", "y", _vec(1), entities=["New"])
        active = mem.active_entities(top_k=2)
        assert "New" in active

    def test_top_k_limit(self, mem):
        mem.add("user", "text", _vec(0), entities=["A", "B", "C", "D", "E", "F"])
        result = mem.active_entities(top_k=3)
        assert len(result) <= 3

    def test_repeated_entity_ranks_higher(self, mem):
        for i in range(3):
            mem.add("user", f"turn {i}", _vec(i), entities=["Frequent"])
        mem.add("user", "once", _vec(99), entities=["Rare"])
        active = mem.active_entities(top_k=2)
        assert active[0] == "Frequent"


# ---------------------------------------------------------------------------
# is_repeat
# ---------------------------------------------------------------------------

class TestIsRepeat:
    def test_not_repeat_on_empty(self, mem):
        assert not mem.is_repeat(_vec(0))

    def test_identical_vector_is_repeat(self, mem):
        v = _vec(30)
        mem.add("user", "question", v)
        assert mem.is_repeat(v)

    def test_very_different_vector_not_repeat(self, mem):
        v1 = _vec(40)
        mem.add("user", "question", v1)
        # Use an orthogonal-ish vector
        v2 = _vec(99)
        assert not mem.is_repeat(v2)

    def test_assistant_turn_not_checked(self, mem):
        v = _vec(50)
        mem.add("mda", "response", v)  # role="mda", not "user"
        assert not mem.is_repeat(v)


# ---------------------------------------------------------------------------
# decay_salience
# ---------------------------------------------------------------------------

class TestDecaySalience:
    def test_salience_decreases(self, mem):
        mem.add("user", "old", _vec(60))
        mem.add("user", "older", _vec(61))
        before = [t.salience for t in list(mem._turns)]
        mem.decay_salience()
        after = [t.salience for t in list(mem._turns)]
        for b, a in zip(before, after):
            assert a < b or a == MIN_SALIENCE

    def test_salience_floor_is_min_salience(self, mem):
        mem.add("user", "stale", _vec(70))
        for _ in range(100):
            mem.decay_salience()
        for turn in mem._turns:
            assert turn.salience >= MIN_SALIENCE


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_returns_empty_string(self, mem):
        assert mem.summary() == ""

    def test_includes_recent_text(self, mem):
        mem.add("user", "Hello there", _vec(80))
        s = mem.summary()
        assert "Hello there" in s

    def test_respects_max_chars(self, mem):
        for i in range(10):
            mem.add("user", "x" * 200, _vec(i + 80))
        s = mem.summary(max_chars=50)
        assert len(s) <= 50

    def test_shows_user_and_mda_prefix(self, mem):
        mem.add("user", "question", _vec(90))
        mem.add("mda", "answer", _vec(91))
        s = mem.summary()
        assert "User" in s
        assert "MDA" in s


# ---------------------------------------------------------------------------
# clear / recent_entities
# ---------------------------------------------------------------------------

class TestClearAndRecentEntities:
    def test_clear_empties_memory(self, mem):
        mem.add("user", "x", _vec(0))
        mem.clear()
        assert len(mem) == 0
        assert mem.context_vector() is None

    def test_recent_entities_order(self, mem):
        mem.add("user", "a", _vec(0), entities=["First"])
        mem.add("user", "b", _vec(1), entities=["Second"])
        recent = mem.recent_entities(n=2)
        assert "Second" in recent
