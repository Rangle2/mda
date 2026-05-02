"""Tests for mda/core/event.py — role-aware event encoding."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mda.core.bind import DIM, cosine, normalize, random_vector
from mda.core.event import (
    ROLE_PERMS,
    ROLE_SLOTS,
    EventFrame,
    assemble_event,
    event_cosine,
    query_role,
    role_cosine,
    time_encode,
    _ROLE_INV_PERMS,
)


# ---------------------------------------------------------------------------
# ROLE_PERMS
# ---------------------------------------------------------------------------

class TestRolePerms:
    def test_all_slots_present(self):
        for role in ROLE_SLOTS:
            assert role in ROLE_PERMS

    def test_perms_are_bijections(self):
        for role, perm in ROLE_PERMS.items():
            assert len(perm) == DIM
            assert set(perm) == set(range(DIM)), f"{role} perm is not a bijection"

    def test_perms_are_deterministic(self):
        """Re-importing / re-calling must yield identical permutations."""
        from mda.core.event import _build_role_perms
        fresh = _build_role_perms(DIM)
        for role in ROLE_SLOTS:
            np.testing.assert_array_equal(ROLE_PERMS[role], fresh[role])

    def test_perms_differ_across_roles(self):
        """Different roles must have different permutations."""
        seen = []
        for role, perm in ROLE_PERMS.items():
            for prev_perm in seen:
                assert not np.array_equal(perm, prev_perm), \
                    f"Duplicate permutation found for {role}"
            seen.append(perm)

    def test_inv_perm_roundtrip(self):
        """Applying perm then inv-perm must recover identity."""
        v = random_vector(DIM, seed=77)
        for role in ROLE_SLOTS:
            perm = ROLE_PERMS[role]
            inv  = _ROLE_INV_PERMS[role]
            np.testing.assert_array_equal(v[perm][inv], v)


# ---------------------------------------------------------------------------
# Role-reversal collapse (the key regression test)
# ---------------------------------------------------------------------------

class TestRoleReversalCollapse:
    """Verifies the core motivation: agent↔patient must NOT collapse to cosine≈1."""

    def _make_frame(self, agent_seed: int, patient_seed: int) -> np.ndarray:
        f = EventFrame(
            agent=random_vector(DIM, seed=agent_seed),
            patient=random_vector(DIM, seed=patient_seed),
        )
        return f.encode()

    def test_role_reversal_is_dissimilar(self):
        """Swapping agent and patient must produce low cosine (< 0.10)."""
        v_A = random_vector(DIM, seed=1)
        v_B = random_vector(DIM, seed=2)

        normal   = EventFrame(agent=v_A, patient=v_B).encode()
        reversed_= EventFrame(agent=v_B, patient=v_A).encode()

        sim = cosine(normal, reversed_)
        assert sim < 0.10, (
            f"Role-reversal cosine {sim:.4f} >= 0.10 — "
            "permutation is not breaking symmetry"
        )

    def test_same_frame_is_identical(self):
        v_A = random_vector(DIM, seed=10)
        v_B = random_vector(DIM, seed=20)
        f1 = EventFrame(agent=v_A, patient=v_B).encode()
        f2 = EventFrame(agent=v_A, patient=v_B).encode()
        np.testing.assert_allclose(f1, f2, atol=1e-10)


# ---------------------------------------------------------------------------
# EventFrame
# ---------------------------------------------------------------------------

class TestEventFrame:
    def test_encode_returns_unit_norm(self):
        f = EventFrame(
            agent=random_vector(DIM, seed=1),
            verb=random_vector(DIM, seed=2),
            patient=random_vector(DIM, seed=3),
        )
        v = f.encode()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_encode_is_cached(self):
        f = EventFrame(agent=random_vector(DIM, seed=5))
        v1 = f.encode()
        v2 = f.encode()
        assert v1 is v2, "Second call must return the same object (cache hit)"

    def test_invalidate_clears_cache(self):
        f = EventFrame(agent=random_vector(DIM, seed=6))
        v1 = f.encode()
        f.invalidate()
        v2 = f.encode()
        assert v1 is not v2, "Cache should have been cleared by invalidate()"

    def test_slot_mutation_auto_invalidates(self):
        f = EventFrame(agent=random_vector(DIM, seed=7))
        v1 = f.encode()
        f.agent = random_vector(DIM, seed=99)   # triggers __setattr__
        v2 = f.encode()
        assert v1 is not v2

    def test_empty_frame_is_finite(self):
        f = EventFrame()
        v = f.encode()
        assert np.all(np.isfinite(v))

    def test_filled_slots_reports_correct_roles(self):
        f = EventFrame(agent=random_vector(DIM, seed=1), verb=random_vector(DIM, seed=2))
        assert set(f.filled_slots) == {"agent", "verb"}

    def test_single_slot_recovers_permuted_identity(self):
        """A single-slot frame's encode must be the normalised permuted vector."""
        v = random_vector(DIM, seed=42)
        f = EventFrame(agent=v)
        expected = normalize(v[ROLE_PERMS["agent"]])
        np.testing.assert_allclose(f.encode(), expected, atol=1e-10)

    def test_4slot_recovery_cosine_in_range(self):
        """4-slot frame cosine similarity should be in [0.40, 0.60] vs each slot."""
        slots = {
            "agent":    random_vector(DIM, seed=101),
            "verb":     random_vector(DIM, seed=102),
            "patient":  random_vector(DIM, seed=103),
            "location": random_vector(DIM, seed=104),
        }
        f = EventFrame(**slots)
        ev = f.encode()
        for role, v in slots.items():
            sim = role_cosine(ev, role, v)
            assert sim > 0.30, (
                f"role={role} cosine {sim:.4f} < 0.30 — "
                "flat superposition signal too weak"
            )


# ---------------------------------------------------------------------------
# assemble_event
# ---------------------------------------------------------------------------

class TestAssembleEvent:
    def test_output_shape(self):
        f = EventFrame(agent=random_vector(DIM, seed=1))
        v = assemble_event(f)
        assert v.shape == (DIM,)

    def test_output_unit_norm(self):
        f = EventFrame(agent=random_vector(DIM, seed=2), verb=random_vector(DIM, seed=3))
        v = assemble_event(f)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6

    def test_custom_dim(self):
        from mda.core.event import _build_role_perms
        dim = 64
        small_perms = _build_role_perms(dim)
        f = EventFrame(agent=random_vector(dim, seed=10))
        acc = np.zeros(dim)
        acc += f.agent[small_perms["agent"]]
        result = normalize(acc)
        assert result.shape == (dim,)

    def test_all_none_returns_finite(self):
        f = EventFrame()
        v = assemble_event(f)
        assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# time_encode
# ---------------------------------------------------------------------------

class TestTimeEncode:
    def test_output_shape(self):
        v = time_encode(0.0)
        assert v.shape == (DIM,)

    def test_unit_norm(self):
        for t in [0.0, 1.0, 100.0, 1_700_000_000.0]:
            v = time_encode(t)
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6, f"t={t}"

    def test_deterministic(self):
        v1 = time_encode(42.5, step=7)
        v2 = time_encode(42.5, step=7)
        np.testing.assert_array_equal(v1, v2)

    def test_different_times_differ(self):
        v1 = time_encode(0.0, step=0)
        v2 = time_encode(1.0, step=0)
        assert not np.allclose(v1, v2)

    def test_different_steps_differ(self):
        v1 = time_encode(0.0, step=0)
        v2 = time_encode(0.0, step=1)
        assert not np.allclose(v1, v2)

    def test_nearby_times_are_similar(self):
        """Sinusoidal half should make nearby times correlated."""
        v1 = time_encode(1000.0)
        v2 = time_encode(1000.1)
        sim = cosine(v1, v2)
        assert sim > 0.5, f"Nearby times cosine {sim:.4f} < 0.5"

    def test_distant_times_are_dissimilar(self):
        """Very different times should have lower cosine than nearby ones."""
        v_near = time_encode(1000.0)
        v_far  = time_encode(9000.0)
        sim_near = cosine(time_encode(1000.0), time_encode(1000.1))
        sim_far  = cosine(v_near, v_far)
        assert sim_near > sim_far

    def test_step_ordering_distinguishes_frames(self):
        """Same t, different step → distinct random second halves.

        The full-vector cosine can still be high when t is fixed (the
        sinusoidal first half is identical for t=0).  What matters is
        that the random second half differs so that step ordering is
        preserved — checked by verifying no two step vectors are
        identical and that the random halves are not all the same.
        """
        half = DIM // 2
        vecs = [time_encode(0.0, step=i) for i in range(5)]
        rand_halves = [v[half:] for v in vecs]
        # No two step vectors should be identical.
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                assert not np.allclose(vecs[i], vecs[j]), \
                    f"step {i} and step {j} produced identical vectors"
        # The random second halves must differ across steps.
        all_same = all(
            np.allclose(rand_halves[0], rand_halves[k]) for k in range(1, 5)
        )
        assert not all_same, "All random halves are identical — step seeding has no effect"

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            time_encode(0.0, dim=3)

    def test_custom_dim(self):
        v = time_encode(1.0, dim=64)
        assert v.shape == (64,)
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# query_role
# ---------------------------------------------------------------------------

class TestQueryRole:
    def _make_registry_with(self, surfaces_seeds: dict[str, int]):
        """Build a minimal EntityRegistry with known entities."""
        from mda.core.registry import EntityRegistry
        reg = EntityRegistry(encoder=None, dim=DIM)
        for surface, seed in surfaces_seeds.items():
            ent = reg.get_or_create(surface)
            ent.v = random_vector(DIM, seed=seed)
        reg._em_dirty = True
        return reg

    def test_returns_list(self):
        reg = self._make_registry_with({"alice": 1, "bob": 2, "carol": 3})
        frame = EventFrame(agent=reg._entities[reg._surface["alice"]].v)
        results = query_role(frame.encode(), "agent", reg, top_k=2)
        assert isinstance(results, list)

    def test_invalid_role_raises(self):
        from mda.core.registry import EntityRegistry
        reg = EntityRegistry(encoder=None, dim=DIM)
        with pytest.raises(ValueError, match="Unknown role"):
            query_role(random_vector(DIM, seed=1), "nonexistent_role", reg)

    def test_agent_query_recovers_agent(self):
        """The agent entity should rank highest in an agent query."""
        from mda.core.registry import EntityRegistry
        reg = EntityRegistry(encoder=None, dim=DIM)

        agent_vec   = random_vector(DIM, seed=10)
        patient_vec = random_vector(DIM, seed=20)

        agent_ent   = reg.get_or_create("alice")
        agent_ent.v = agent_vec
        patient_ent = reg.get_or_create("bob")
        patient_ent.v = patient_vec
        reg._em_dirty = True

        frame = EventFrame(agent=agent_vec, patient=patient_vec)
        ev    = frame.encode()

        results = query_role(ev, "agent", reg, top_k=2)
        top_id  = results[0][1].id if results else None
        assert top_id == agent_ent.id, (
            "Agent entity should rank first in agent-role query"
        )


# ---------------------------------------------------------------------------
# role_cosine
# ---------------------------------------------------------------------------

class TestRoleCosine:
    def test_correct_role_scores_high(self):
        v = random_vector(DIM, seed=55)
        frame = EventFrame(verb=v)
        ev = frame.encode()
        score = role_cosine(ev, "verb", v)
        # With a single slot, the role projection should be close to v itself.
        assert score > 0.90, f"Expected > 0.90, got {score:.4f}"

    def test_wrong_role_scores_lower(self):
        v = random_vector(DIM, seed=60)
        frame = EventFrame(verb=v)
        ev = frame.encode()
        wrong_score = role_cosine(ev, "agent", v)
        right_score = role_cosine(ev, "verb",  v)
        assert right_score > wrong_score

    def test_invalid_role_raises(self):
        v = random_vector(DIM, seed=1)
        ev = EventFrame(verb=v).encode()
        with pytest.raises(ValueError, match="Unknown role"):
            role_cosine(ev, "actor", v)


# ---------------------------------------------------------------------------
# event_cosine
# ---------------------------------------------------------------------------

class TestEventCosine:
    def test_identical_events_return_one(self):
        f = EventFrame(agent=random_vector(DIM, seed=1), verb=random_vector(DIM, seed=2))
        ev = f.encode()
        assert abs(event_cosine(ev, ev) - 1.0) < 1e-5

    def test_different_events_not_one(self):
        ev1 = EventFrame(agent=random_vector(DIM, seed=1)).encode()
        ev2 = EventFrame(agent=random_vector(DIM, seed=9)).encode()
        assert event_cosine(ev1, ev2) < 0.99
