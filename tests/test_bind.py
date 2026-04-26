"""Tests for core/bind.py — vector primitives."""
import numpy as np
import pytest
from core.bind import (
    DIM, normalize, random_vector, zero_vector,
    bind, unbind, bind_many, cosine,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_unit_length(self):
        v = np.array([3.0, 4.0])
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_zero_vector_safe(self):
        v = np.zeros(DIM)
        n = normalize(v)
        assert np.all(np.isfinite(n))

    def test_already_normalized(self):
        v = normalize(np.ones(DIM))
        n = normalize(v)
        np.testing.assert_allclose(n, v, atol=1e-6)

    def test_negative_values(self):
        v = np.array([-1.0, 0.0, 0.0])
        n = normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# random_vector
# ---------------------------------------------------------------------------

class TestRandomVector:
    def test_unit_norm(self):
        v = random_vector(DIM, seed=42)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_deterministic_seed(self):
        v1 = random_vector(DIM, seed=7)
        v2 = random_vector(DIM, seed=7)
        np.testing.assert_array_equal(v1, v2)

    def test_different_seeds_differ(self):
        v1 = random_vector(DIM, seed=1)
        v2 = random_vector(DIM, seed=2)
        assert not np.allclose(v1, v2)

    def test_custom_dim(self):
        v = random_vector(dim=64, seed=0)
        assert v.shape == (64,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# zero_vector
# ---------------------------------------------------------------------------

class TestZeroVector:
    def test_all_zeros(self):
        v = zero_vector(DIM)
        assert np.all(v == 0.0)

    def test_default_dim(self):
        assert zero_vector().shape == (DIM,)

    def test_custom_dim(self):
        assert zero_vector(32).shape == (32,)


# ---------------------------------------------------------------------------
# bind / unbind
# ---------------------------------------------------------------------------

class TestBind:
    def test_bind_returns_same_dim(self):
        a = random_vector(DIM, seed=1)
        b = random_vector(DIM, seed=2)
        c = bind(a, b)
        assert c.shape == (DIM,)

    def test_bind_is_approximately_commutative(self):
        a = random_vector(DIM, seed=1)
        b = random_vector(DIM, seed=2)
        np.testing.assert_allclose(bind(a, b), bind(b, a), atol=1e-10)

    def test_unbind_recovers_original(self):
        a = random_vector(DIM, seed=3)
        b = random_vector(DIM, seed=4)
        compound = bind(a, b)
        recovered = unbind(compound, b)
        # Should be highly similar to a (not identical due to HRR properties)
        sim = cosine(normalize(recovered), normalize(a))
        assert sim > 0.99

    def test_bind_distinct_from_inputs(self):
        a = random_vector(DIM, seed=5)
        b = random_vector(DIM, seed=6)
        c = bind(a, b)
        assert not np.allclose(c, a)
        assert not np.allclose(c, b)


# ---------------------------------------------------------------------------
# bind_many
# ---------------------------------------------------------------------------

class TestBindMany:
    def test_single_input(self):
        a = random_vector(DIM, seed=10)
        result = bind_many(a)
        np.testing.assert_array_equal(result, a)

    def test_two_inputs_matches_bind(self):
        a = random_vector(DIM, seed=11)
        b = random_vector(DIM, seed=12)
        np.testing.assert_allclose(bind_many(a, b), bind(a, b), atol=1e-10)

    def test_three_inputs_shape(self):
        vecs = [random_vector(DIM, seed=i) for i in range(3)]
        result = bind_many(*vecs)
        assert result.shape == (DIM,)

    def test_does_not_mutate_inputs(self):
        a = random_vector(DIM, seed=20)
        b = random_vector(DIM, seed=21)
        a_copy = a.copy()
        bind_many(a, b)
        np.testing.assert_array_equal(a, a_copy)


# ---------------------------------------------------------------------------
# cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_vectors_returns_one(self):
        v = random_vector(DIM, seed=30)
        assert abs(cosine(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors_near_zero(self):
        a = np.zeros(DIM)
        b = np.zeros(DIM)
        a[0] = 1.0
        b[1] = 1.0
        assert abs(cosine(a, b)) < 1e-5

    def test_opposite_vectors_near_neg_one(self):
        v = random_vector(DIM, seed=40)
        assert abs(cosine(v, -v) + 1.0) < 1e-5

    def test_range(self):
        a = random_vector(DIM, seed=50)
        b = random_vector(DIM, seed=51)
        s = cosine(a, b)
        assert -1.0 <= s <= 1.0

    def test_zero_vector_safe(self):
        a = np.zeros(DIM)
        b = random_vector(DIM, seed=52)
        s = cosine(a, b)
        assert np.isfinite(s)
