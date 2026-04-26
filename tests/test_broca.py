"""Tests for inference/broca.py — BrocaModule."""
import numpy as np
import pytest
from core.bind import DIM, normalize, random_vector
from core.entity import Entity
from inference.broca import BrocaModule


# ---------------------------------------------------------------------------
# _encode / cache
# ---------------------------------------------------------------------------

class TestEncodeCache:
    def test_returns_normalized_vector(self, broca):
        v = broca._encode("hello")
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_cache_hit_returns_same_object(self, broca):
        v1 = broca._encode("cached text")
        v2 = broca._encode("cached text")
        assert v1 is v2

    def test_different_texts_differ(self, broca):
        v1 = broca._encode("alpha")
        v2 = broca._encode("beta")
        assert not np.allclose(v1, v2)

    def test_cache_eviction_at_limit(self, broca):
        # Fill cache beyond 2000 entries
        for i in range(2001):
            broca._encode(f"unique text {i}")
        # After eviction, a new entry should still work
        v = broca._encode("after eviction")
        assert v.shape == (DIM,)

    def test_cache_cleared_on_eviction(self, broca):
        for i in range(2001):
            broca._encode(f"entry {i}")
        # Cache holds 2001 entries; the *next* call triggers clear + one new entry
        broca._encode("trigger eviction")
        assert len(broca._encode_cache) == 1


# ---------------------------------------------------------------------------
# store_facts / learn_from_facts / vocab_size
# ---------------------------------------------------------------------------

class TestStoreFacts:
    def test_stores_new_facts(self, broca, registry):
        e = registry.get_or_create("Python", "software")
        broca.store_facts(e.id, ["Python is a programming language."])
        assert e.id in broca._entity_facts
        assert len(broca._entity_facts[e.id]) == 1

    def test_deduplication(self, broca, registry):
        e = registry.get_or_create("Python", "software")
        fact = "Python is interpreted."
        broca.store_facts(e.id, [fact])
        broca.store_facts(e.id, [fact])
        assert broca._entity_facts[e.id].count(fact) == 1

    def test_junk_filtered_out(self, broca, registry):
        e = registry.get_or_create("Unknown", "unknown")
        broca.store_facts(e.id, ["I don't have enough information about this."])
        assert e.id not in broca._entity_facts or \
               len(broca._entity_facts.get(e.id, [])) == 0

    def test_empty_list_noop(self, broca, registry):
        e = registry.get_or_create("Empty", "unknown")
        broca.store_facts(e.id, [])
        assert e.id not in broca._entity_facts

    def test_fact_vecs_created(self, broca, registry):
        e = registry.get_or_create("Java", "software")
        broca.store_facts(e.id, ["Java runs on the JVM."])
        assert e.id in broca._fact_vecs
        assert len(broca._fact_vecs[e.id]) == 1

    def test_learn_from_facts_alias(self, broca, registry):
        e = registry.get_or_create("Rust", "software")
        broca.learn_from_facts(e.id, ["Rust guarantees memory safety."])
        assert e.id in broca._entity_facts

    def test_vocab_size_increments(self, broca, registry):
        before = broca.vocab_size()["fact_store"]
        e = registry.get_or_create("Go", "software")
        broca.store_facts(e.id, ["Go was created at Google."])
        after = broca.vocab_size()["fact_store"]
        assert after > before


# ---------------------------------------------------------------------------
# _score_facts
# ---------------------------------------------------------------------------

class TestScoreFacts:
    def test_returns_empty_for_unknown_entity(self, broca, entity):
        q = broca._encode("query")
        result = broca._score_facts(entity, q, top_k=3)
        assert result == []

    def test_returns_top_k(self, broca, registry):
        e = registry.get_or_create("Scored", "concept")
        facts = [f"fact number {i}" for i in range(5)]
        broca.store_facts(e.id, facts)
        q = broca._encode("fact")
        result = broca._score_facts(e, q, top_k=2)
        assert len(result) <= 2

    def test_result_sorted_descending(self, broca, registry):
        e = registry.get_or_create("Sorted", "concept")
        broca.store_facts(e.id, ["alpha is great", "beta is fine", "gamma is okay"])
        q = broca._encode("alpha")
        result = broca._score_facts(e, q, top_k=3)
        scores = [r[0] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_result_tuple_structure(self, broca, registry):
        e = registry.get_or_create("Structured", "concept")
        broca.store_facts(e.id, ["some fact about structured entity"])
        q = broca._encode("structured")
        result = broca._score_facts(e, q, top_k=1)
        assert len(result) == 1
        score, fact = result[0]
        assert isinstance(score, float)
        assert isinstance(fact, str)


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_unknown_entity_returns_zero(self, broca):
        score = broca.confidence("nonexistent_id", "query")
        assert score == 0.0

    def test_entity_without_facts_returns_zero(self, broca, registry):
        e = registry.get_or_create("Factless", "unknown")
        score = broca.confidence(e.id, "query")
        assert score == 0.0

    def test_entity_with_facts_returns_positive(self, broca, registry):
        e = registry.get_or_create("Knowledgeable", "concept")
        broca.store_facts(e.id, ["Knowledgeable is a well-known concept in ML."])
        score = broca.confidence(e.id, "Knowledgeable concept")
        assert score > 0.0

    def test_result_clamped_between_0_and_1(self, broca, registry):
        e = registry.get_or_create("Clamped", "concept")
        broca.store_facts(e.id, ["Clamped entity has facts."])
        score = broca.confidence(e.id, "clamped")
        assert 0.0 <= score <= 1.0

    def test_relevant_query_scores_higher(self, broca, registry):
        e = registry.get_or_create("Database", "software")
        broca.store_facts(e.id, ["Database stores and retrieves structured data."])
        relevant_score = broca.confidence(e.id, "database storage")
        irrelevant_score = broca.confidence(e.id, "banana smoothie recipe")
        assert relevant_score >= irrelevant_score


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_no_facts_returns_default(self, broca, entity):
        result = broca.generate(entity)
        assert "Not enough information" in result

    def test_with_facts_returns_string(self, broca, registry):
        e = registry.get_or_create("Generated", "concept")
        broca.store_facts(e.id, [
            "Generated entity is used in testing.",
            "Generated entity helps verify output.",
        ])
        result = broca.generate(e, query_type="nedir")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_list_query_type_bullet_format(self, broca, registry):
        e = registry.get_or_create("Listed", "concept")
        broca.store_facts(e.id, [
            "Listed entity was created in 2024.",
            "Listed entity runs on Python.",
        ])
        result = broca.generate(e, query_type="list")
        assert "•" in result or "Known facts" in result

    def test_custom_query_text(self, broca, registry):
        e = registry.get_or_create("Custom", "concept")
        broca.store_facts(e.id, ["Custom entity does custom things."])
        result = broca.generate(e, query_type="nedir", query_text="what is this?")
        assert isinstance(result, str)
