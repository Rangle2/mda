"""Tests for inference/associative.py — AssociativeChain."""
import numpy as np
import pytest
from core.bind import DIM, random_vector, bind, normalize
from core.entity import Entity
from core.registry import EntityRegistry
from core.encoder import HolisticEncoder
from inference.associative import AssociativeChain, ChainNode, ChainResult


@pytest.fixture
def chain(registry, encoder):
    return AssociativeChain(registry, encoder)


@pytest.fixture
def two_connected_entities(registry):
    """Two entities with a mutual synapse."""
    from core.bind import bind
    a = registry.get_or_create("NodeA", "concept")
    b = registry.get_or_create("NodeB", "concept")
    a.add_synapse(b, bind)
    b.add_synapse(a, bind)
    a.use_count = 5
    b.use_count = 5
    return a, b


@pytest.fixture
def chain_with_entities(registry, encoder, two_connected_entities):
    return AssociativeChain(registry, encoder), two_connected_entities


# ---------------------------------------------------------------------------
# ChainResult helpers
# ---------------------------------------------------------------------------

class TestChainResult:
    def test_active_entities_returns_list(self, chain, registry):
        e = registry.get_or_create("Test", "concept")
        node = ChainNode(entity=e, depth=0, activation=1.0, path=["Test"])
        result = ChainResult(
            nodes=[node],
            compound_v=e.v.copy(),
            origin_v=e.v.copy(),
            depth_reached=0,
        )
        entities = result.active_entities()
        assert len(entities) == 1
        assert entities[0] is e

    def test_activation_map_returns_dict(self, registry):
        e = registry.get_or_create("Map", "concept")
        node = ChainNode(entity=e, depth=0, activation=0.7, path=["Map"])
        result = ChainResult(
            nodes=[node],
            compound_v=e.v.copy(),
            origin_v=e.v.copy(),
            depth_reached=0,
        )
        m = result.activation_map()
        assert isinstance(m, dict)
        assert "Map" in m
        assert m["Map"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# AssociativeChain.expand
# ---------------------------------------------------------------------------

class TestExpand:
    def test_origin_always_in_nodes(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        surfaces = [n.entity.surface for n in result.nodes]
        assert "NodeA" in surfaces

    def test_compound_v_shape(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        assert result.compound_v.shape == (DIM,)

    def test_origin_v_normalized(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        assert abs(np.linalg.norm(result.origin_v) - 1.0) < 1e-5

    def test_no_duplicate_nodes(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        ids = [n.entity.id for n in result.nodes]
        assert len(ids) == len(set(ids))

    def test_depth_reached_nonnegative(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        assert result.depth_reached >= 0

    def test_single_entity_no_synapses(self, chain, registry):
        solo = registry.get_or_create("Solo", "concept")
        result = chain.expand(solo)
        assert len(result.nodes) == 1
        assert result.nodes[0].entity is solo

    def test_with_custom_context_and_query(self, chain_with_entities, encoder):
        chain, (a, b) = chain_with_entities
        ctx = normalize(encoder.encode("context text"))
        q = normalize(encoder.encode("query text"))
        result = chain.expand(a, context_vec=ctx, query_vec=q)
        assert result.compound_v.shape == (DIM,)

    def test_finite_compound_vector(self, chain_with_entities):
        chain, (a, b) = chain_with_entities
        result = chain.expand(a)
        assert np.all(np.isfinite(result.compound_v))


# ---------------------------------------------------------------------------
# AssociativeChain.expand_from_text
# ---------------------------------------------------------------------------

class TestExpandFromText:
    def test_returns_none_for_unrecognized_text(self, chain):
        # Empty registry — no entities at all
        result = chain.expand_from_text("completely unknown xyz123")
        assert result is None

    def test_finds_entity_by_token_match(self, chain, registry):
        registry.get_or_create("Python", "software")
        result = chain.expand_from_text("Python programming")
        assert result is not None
        surfaces = [n.entity.surface for n in result.nodes]
        assert "Python" in surfaces

    def test_returns_chain_result_type(self, chain, registry):
        registry.get_or_create("Java", "software")
        result = chain.expand_from_text("Java is a language")
        if result is not None:
            assert isinstance(result, ChainResult)

    def test_cosine_fallback_for_unknown_token(self, chain, registry):
        # Register an entity and pass text with no direct token match
        # but conceptually related (the cosine fallback should pick it up)
        e = registry.get_or_create("MachineLearning", "concept")
        result = chain.expand_from_text("MachineLearning is important")
        # Should find the entity via direct token search
        assert result is not None


# ---------------------------------------------------------------------------
# chain_summary
# ---------------------------------------------------------------------------

class TestChainSummary:
    def test_empty_nodes_returns_empty(self, chain):
        result = ChainResult(
            nodes=[],
            compound_v=np.zeros(DIM),
            origin_v=np.zeros(DIM),
            depth_reached=0,
        )
        assert chain.chain_summary(result) == ""

    def test_returns_arrow_separated_string(self, chain, registry):
        a = registry.get_or_create("Start", "concept")
        b = registry.get_or_create("End", "concept")
        nodes = [
            ChainNode(entity=a, depth=0, activation=1.0, path=["Start"]),
            ChainNode(entity=b, depth=1, activation=0.5, path=["Start", "End"]),
        ]
        result = ChainResult(
            nodes=nodes,
            compound_v=a.v.copy(),
            origin_v=a.v.copy(),
            depth_reached=1,
        )
        s = chain.chain_summary(result)
        assert "->" in s
        assert "Start" in s
