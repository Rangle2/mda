"""Tests for inference/reasoning.py — ReasoningEngine."""
import numpy as np
import pytest
from core.bind import DIM, random_vector, normalize
from core.encoder import HolisticEncoder
from core.registry import EntityRegistry
from inference.reasoning import ReasoningEngine, ThoughtStep


@pytest.fixture
def engine(encoder, registry):
    return ReasoningEngine(encoder, registry)


@pytest.fixture
def populated_engine(encoder, registry):
    """Engine with two entities that have trained W matrices."""
    a = registry.get_or_create("Neuron", "software")
    b = registry.get_or_create("Learning", "concept")
    target = normalize(random_vector(DIM, seed=0))
    for _ in range(5):
        a.update_W(target)
        b.update_W(target)
    return ReasoningEngine(encoder, registry), a, b


# ---------------------------------------------------------------------------
# compose / decompose
# ---------------------------------------------------------------------------

class TestCompose:
    def test_single_surface_returns_entity_vector(self, engine, registry):
        e = registry.get_or_create("Alpha", "concept")
        result = engine.compose("Alpha")
        np.testing.assert_array_equal(result, e.v)

    def test_unknown_surface_uses_encoder(self, engine):
        result = engine.compose("unknownsurface123")
        assert result.shape == (DIM,)
        assert np.all(np.isfinite(result))

    def test_two_surfaces_returns_dim_vector(self, engine, registry):
        registry.get_or_create("A", "concept")
        registry.get_or_create("B", "concept")
        result = engine.compose("A", "B")
        assert result.shape == (DIM,)

    def test_composed_differs_from_inputs(self, engine, registry):
        a = registry.get_or_create("X", "concept")
        b = registry.get_or_create("Y", "concept")
        result = engine.compose("X", "Y")
        assert not np.allclose(result, a.v)
        assert not np.allclose(result, b.v)

    def test_no_surfaces_returns_zeros(self, engine):
        result = engine.compose()
        np.testing.assert_array_equal(result, np.zeros(DIM))


class TestDecompose:
    def test_decompose_recovers_approximate_original(self, engine, registry):
        registry.get_or_create("P", "concept")
        registry.get_or_create("Q", "concept")
        composed = engine.compose("P", "Q")
        recovered = engine.decompose(composed, "Q")
        from core.bind import cosine
        p_entity = registry.get("P")
        sim = cosine(normalize(recovered), normalize(p_entity.v))
        assert sim > 0.9

    def test_decompose_returns_same_dim(self, engine, registry):
        registry.get_or_create("M", "concept")
        v = random_vector(DIM, seed=0)
        result = engine.decompose(v, "M")
        assert result.shape == (DIM,)


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------

class TestRefine:
    def test_unknown_entity_returns_input(self, engine):
        v = random_vector(DIM, seed=0)
        result, trace = engine.refine(v, "DoesNotExist")
        np.testing.assert_array_equal(result, v)
        assert trace == []

    def test_returns_vector_and_trace(self, populated_engine):
        eng, a, b = populated_engine
        v = random_vector(DIM, seed=1)
        result, trace = eng.refine(v, "Neuron")
        assert result.shape == (DIM,)
        assert isinstance(trace, list)

    def test_trace_contains_thought_steps(self, populated_engine):
        eng, a, b = populated_engine
        v = random_vector(DIM, seed=2)
        _, trace = eng.refine(v, "Neuron")
        for step in trace:
            assert isinstance(step, ThoughtStep)

    def test_with_query_vec(self, populated_engine):
        eng, a, b = populated_engine
        v = random_vector(DIM, seed=3)
        q = random_vector(DIM, seed=4)
        result, _ = eng.refine(v, "Neuron", query_vec=q)
        assert result.shape == (DIM,)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# reason
# ---------------------------------------------------------------------------

class TestReason:
    def test_returns_vector_and_trace(self, populated_engine):
        eng, a, b = populated_engine
        result, trace = eng.reason("what is Neuron?", ["Neuron"])
        assert result.shape == (DIM,)
        assert isinstance(trace, list)

    def test_empty_entity_list_picks_best(self, populated_engine):
        eng, a, b = populated_engine
        result, trace = eng.reason("Neuron Learning", [])
        assert result.shape == (DIM,)

    def test_multiple_entities(self, populated_engine):
        eng, a, b = populated_engine
        result, _ = eng.reason("Neuron and Learning", ["Neuron", "Learning"])
        assert result.shape == (DIM,)
        assert np.all(np.isfinite(result))

    def test_empty_registry_returns_query_vec(self, engine):
        result, trace = engine.reason("anything", [])
        assert result.shape == (DIM,)


# ---------------------------------------------------------------------------
# thought_trace_str
# ---------------------------------------------------------------------------

class TestThoughtTraceStr:
    def test_empty_trace_returns_empty_string(self, engine):
        assert engine.thought_trace_str([]) == ""

    def test_returns_string(self, populated_engine):
        eng, a, b = populated_engine
        v = random_vector(DIM, seed=0)
        _, trace = eng.refine(v, "Neuron")
        s = eng.thought_trace_str(trace)
        assert isinstance(s, str)

    def test_contains_step_markers(self, populated_engine):
        eng, a, b = populated_engine
        v = random_vector(DIM, seed=0)
        _, trace = eng.refine(v, "Neuron")
        if trace:
            s = eng.thought_trace_str(trace)
            assert "[" in s


# ---------------------------------------------------------------------------
# infer_from_chain
# ---------------------------------------------------------------------------

class TestInferFromChain:
    def test_empty_nodes_returns_empty(self, engine, broca):
        q = random_vector(DIM, seed=0)
        result = engine.infer_from_chain([], q, broca)
        assert result == []

    def test_no_depth2_nodes_returns_empty(self, engine, registry, broca):
        from inference.associative import ChainNode
        e = registry.get_or_create("Solo", "concept")
        node = ChainNode(entity=e, depth=1, activation=1.0, path=["Solo"])
        q = random_vector(DIM, seed=0)
        result = engine.infer_from_chain([node], q, broca)
        assert result == []
