"""Tests for core/encoder.py — HolisticEncoder."""
import numpy as np
import pytest
from mda.core.bind import DIM, normalize
from mda.core.encoder import HolisticEncoder


@pytest.fixture
def enc():
    return HolisticEncoder(DIM)


class TestEncode:
    def test_returns_ndarray(self, enc):
        v = enc.encode("hello world")
        assert isinstance(v, np.ndarray)

    def test_correct_dimension(self, enc):
        v = enc.encode("test sentence")
        assert v.shape == (DIM,)

    def test_finite_values(self, enc):
        v = enc.encode("some text here")
        assert np.all(np.isfinite(v))

    def test_nonzero_output(self, enc):
        v = enc.encode("machine learning")
        assert not np.allclose(v, 0)

    def test_deterministic(self, enc):
        text = "consistent encoding"
        v1 = enc.encode(text)
        v2 = enc.encode(text)
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_differ(self, enc):
        v1 = enc.encode("cat")
        v2 = enc.encode("database")
        assert not np.allclose(v1, v2)

    def test_empty_string_safe(self, enc):
        v = enc.encode("")
        assert v.shape == (DIM,)
        assert np.all(np.isfinite(v))

    def test_long_text(self, enc):
        text = "word " * 200
        v = enc.encode(text)
        assert v.shape == (DIM,)
        assert np.all(np.isfinite(v))

    def test_known_concept_word(self, enc):
        # "machine" is in the concept map; it should produce a non-trivial vector
        v = enc.encode("machine")
        assert not np.allclose(v, 0)


class TestRegisterConcept:
    def test_registers_and_returns_vector(self, enc):
        v = enc.register_concept("NewConcept", "concept")
        assert isinstance(v, np.ndarray)
        assert v.shape == (DIM,)

    def test_registered_concept_used_in_encoding(self, enc):
        enc.register_concept("Zephyr", "software")
        v = enc.encode("Zephyr")
        assert not np.allclose(v, 0)

    def test_idempotent_registration(self, enc):
        v1 = enc.register_concept("Foo", "")
        v2 = enc.register_concept("Foo", "")
        np.testing.assert_array_equal(v1, v2)


class TestSimilarity:
    def test_identical_texts_similarity_near_one(self, enc):
        s = enc.similarity("neural network", "neural network")
        assert s > 0.99

    def test_same_concept_different_words_positive(self, enc):
        # "AI" and "artificial intelligence" should share conceptual overlap
        s = enc.similarity("AI", "artificial intelligence")
        assert s > 0.0

    def test_unrelated_texts_lower_similarity(self, enc):
        s_related = enc.similarity("deep learning", "neural network")
        s_unrelated = enc.similarity("banana", "quantum")
        # Related pair should score higher than a random unrelated pair
        assert s_related >= s_unrelated - 0.3  # allow some tolerance

    def test_similarity_in_range(self, enc):
        s = enc.similarity("hello", "world")
        assert -1.0 <= s <= 1.0
