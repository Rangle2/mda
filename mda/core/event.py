"""
mda/core/event.py — Role-aware event encoding via flat superposition.

Design rationale
----------------
Nested binding collapses to noise at N > 3 elements and, critically,
produces cosine ≈ 1.0 between role-reversed pairs (agent↔patient),
making them indistinguishable.

Flat superposition + per-role permutation fixes both problems:

  event_vec = normalize( Σ_{role r} P_r · slot_r )

where P_r is a deterministic permutation matrix for role r.  Because
each P_r is an independent bijection over the 512 basis dimensions,
role vectors are near-orthogonal after permutation, so:

  cosine(P_agent · v_A + P_patient · v_B,
         P_agent · v_B + P_patient · v_A)  ≈ 0.02

instead of the 1.0 you get with nested bind.

Benchmark (4-slot event recovery): cosine 0.45–0.54, versus ~0.0 with
nested bind.  Cross-lingual verb convergence (çalışır ~ works_at):
cosine → 1.0 after 150 co-occurrence steps with no fixed mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from mda.core.bind import DIM, cosine, normalize

if TYPE_CHECKING:  # avoid circular import at runtime
    from mda.core.registry import EntityRegistry

# ---------------------------------------------------------------------------
# Role slots and permutations
# ---------------------------------------------------------------------------

#: Canonical ordered role slots for a thematic-role event frame.
ROLE_SLOTS: tuple[str, ...] = (
    "agent",
    "verb",
    "patient",
    "location",
    "instrument",
    "cause",
    "result",
    "time",
)

#: Namespace seed used to derive per-role permutation seeds.
#: Change only breaks serialisation compatibility, not correctness.
_ROLE_BASE_SEED: int = 0xEF7A_3B12

# Build once at import time — these are module-level constants.
def _build_role_perms(dim: int = DIM) -> dict[str, np.ndarray]:
    """Return a deterministic permutation array (length *dim*) per role slot."""
    perms: dict[str, np.ndarray] = {}
    for idx, role in enumerate(ROLE_SLOTS):
        seed = (_ROLE_BASE_SEED + idx * 0x9E37_79B9) & 0xFFFF_FFFF
        rng = np.random.default_rng(seed=seed)
        perms[role] = rng.permutation(dim).astype(np.intp)
    return perms


ROLE_PERMS: dict[str, np.ndarray] = _build_role_perms(DIM)

# Inverse permutations — needed by query_role().
_ROLE_INV_PERMS: dict[str, np.ndarray] = {
    role: np.argsort(perm).astype(np.intp)
    for role, perm in ROLE_PERMS.items()
}


# ---------------------------------------------------------------------------
# EventFrame dataclass
# ---------------------------------------------------------------------------

@dataclass
class EventFrame:
    """Structured event memory with optional thematic-role slots.

    Each slot holds a *normalised* 512-dim vector (or None if unknown).
    The assembled event vector is computed lazily on first access and
    cached until ``invalidate()`` is called.

    Example
    -------
    >>> from mda.core.bind import random_vector
    >>> frame = EventFrame(
    ...     agent=random_vector(seed=1),
    ...     verb=random_vector(seed=2),
    ...     patient=random_vector(seed=3),
    ... )
    >>> vec = frame.encode()   # (512,) float64, unit norm
    >>> frame.agent = random_vector(seed=99)
    >>> frame.invalidate()     # must call after mutating a slot
    >>> vec2 = frame.encode()  # recomputed
    """

    agent:      Optional[np.ndarray] = None
    verb:       Optional[np.ndarray] = None
    patient:    Optional[np.ndarray] = None
    location:   Optional[np.ndarray] = None
    instrument: Optional[np.ndarray] = None
    cause:      Optional[np.ndarray] = None
    result:     Optional[np.ndarray] = None
    time:       Optional[np.ndarray] = None

    # Internal lazy cache — excluded from __init__, __repr__, and __eq__.
    _vec: Optional[np.ndarray] = field(
        default=None, repr=False, init=False, compare=False, hash=False
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Discard the cached event vector (call after mutating any slot)."""
        self._vec = None

    def encode(self) -> np.ndarray:
        """Return the assembled, normalised event vector (cached)."""
        if self._vec is None:
            self._vec = assemble_event(self)
        return self._vec

    @property
    def filled_slots(self) -> list[str]:
        """Names of the slots that are not None."""
        return [r for r in ROLE_SLOTS if getattr(self, r) is not None]

    def retrieval_vec(self) -> np.ndarray:
        """Un-permuted slot superposition for text-query relevance gating.

        Unlike ``encode()`` (which applies per-role permutations to preserve
        role identity), this sums slot vectors *without* permutation so they
        remain in the same semantic space as text vectors produced by
        ``HolisticEncoder``.  Use this for cosine gating against a query;
        use ``encode()`` for role-disambiguation queries via ``query_role()``.
        """
        acc: np.ndarray | None = None
        for role in ROLE_SLOTS:
            v = getattr(self, role)
            if v is not None:
                acc = v.copy() if acc is None else acc + v
        if acc is None:
            return np.zeros(DIM, dtype=np.float64)
        return normalize(acc)

    def __setattr__(self, name: str, value: object) -> None:  # noqa: D105
        super().__setattr__(name, value)
        # Invalidate cache whenever a role slot is mutated externally.
        if name in ROLE_SLOTS:
            # Use object.__setattr__ to avoid infinite recursion.
            object.__setattr__(self, "_vec", None)


# ---------------------------------------------------------------------------
# Assembly — flat superposition
# ---------------------------------------------------------------------------

def assemble_event(
    frame: EventFrame,
    dim: int = DIM,
) -> np.ndarray:
    """Assemble a role-aware event vector from *frame* via flat superposition.

    For each filled slot *r* with vector *v_r*:
        contribution = v_r[ ROLE_PERMS[r] ]   # permute into role sub-space

    Result:
        normalize( Σ contributions )

    An all-None frame returns the zero vector (handled gracefully by
    ``normalize``'s ``+ 1e-8`` denominator guard).
    """
    acc = np.zeros(dim, dtype=np.float64)
    for role in ROLE_SLOTS:
        v = getattr(frame, role)
        if v is not None:
            perm = ROLE_PERMS[role]
            acc += v[perm]
    return normalize(acc)


# ---------------------------------------------------------------------------
# Temporal encoding
# ---------------------------------------------------------------------------

def time_encode(
    t: float,
    dim: int = DIM,
    *,
    step: Optional[int] = None,
) -> np.ndarray:
    """Encode a time value into a ``dim``-dimensional vector.

    The vector is split into two halves with complementary properties:

    **First half** (``dim // 2`` dimensions) — sinusoidal:
        Uses a log-spaced frequency ladder (identical to the Transformer
        positional encoding scheme).  Nearby time values produce highly
        similar vectors → supports range queries and temporal proximity.

    **Second half** (``dim // 2`` dimensions) — deterministic random:
        Seeded by ``step`` (if provided) or a quantised hash of ``t``.
        Each distinct step index produces a near-orthogonal vector →
        supports exact step ordering in agentic chains.

    Parameters
    ----------
    t:
        Continuous time value (e.g. Unix timestamp, normalised age).
    dim:
        Output dimensionality (must be even).  Defaults to ``DIM`` (512).
    step:
        Integer step counter.  When provided, the second half is seeded
        by ``step`` for exact ordering; otherwise seeded by a hash of ``t``.

    Returns
    -------
    np.ndarray
        Unit-norm vector of shape ``(dim,)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    half = dim // 2

    # -- Sinusoidal half -------------------------------------------------------
    sin_part = np.empty(half, dtype=np.float64)
    for k in range(half // 2):
        freq = 1.0 / math.pow(10_000.0, 2.0 * k / half)
        angle = t * freq
        sin_part[2 * k]     = math.sin(angle)
        sin_part[2 * k + 1] = math.cos(angle)
    # Handle odd half gracefully (half is odd when dim//2 is odd).
    if half % 2 != 0:
        sin_part[-1] = math.sin(t)

    # -- Deterministic-random half ---------------------------------------------
    if step is not None:
        seed = int(step) & 0xFFFF_FFFF
    else:
        # Quantise t to ~microsecond resolution and fold into 32-bit seed.
        seed = int(abs(t) * 1_000_000) % (2 ** 32)

    rng = np.random.default_rng(seed=seed)
    rand_part = rng.standard_normal(half).astype(np.float64)
    rand_part = normalize(rand_part)

    vec = np.concatenate([sin_part, rand_part])
    return normalize(vec)


# ---------------------------------------------------------------------------
# Role query
# ---------------------------------------------------------------------------

def query_role(
    event_vec: np.ndarray,
    role: str,
    registry: "EntityRegistry",
    top_k: int = 5,
) -> list:
    """Retrieve the best-matching entities for *role* from *event_vec*.

    The inverse permutation P_r⁻¹ projects the event superposition back
    into the original semantic space for role *r*, then the result is
    looked up in *registry* via cosine nearest-neighbour search.

    Parameters
    ----------
    event_vec:
        Assembled event vector (output of :func:`assemble_event`).
    role:
        One of the :data:`ROLE_SLOTS` strings.
    registry:
        The live :class:`~mda.core.registry.EntityRegistry` instance.
    top_k:
        How many candidate entities to return.

    Returns
    -------
    list
        The registry's nearest-neighbour result (list of ``(entity, score)``
        tuples, same format as ``registry.nearest()``).

    Raises
    ------
    ValueError
        If *role* is not a recognised role slot name.
    """
    if role not in ROLE_PERMS:
        raise ValueError(
            f"Unknown role {role!r}. Valid roles: {list(ROLE_SLOTS)}"
        )
    inv_perm = _ROLE_INV_PERMS[role]
    role_query = event_vec[inv_perm]
    return registry.nearest(role_query, top_k)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def event_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two assembled event vectors."""
    return cosine(a, b)


def role_cosine(
    event_vec: np.ndarray,
    role: str,
    candidate: np.ndarray,
) -> float:
    """Cosine similarity between the *role* projection and a candidate vector.

    Useful for scoring a specific entity against a role slot without a
    full registry lookup.

    Parameters
    ----------
    event_vec:
        Assembled event vector.
    role:
        Target role slot name.
    candidate:
        The entity vector to score.
    """
    if role not in _ROLE_INV_PERMS:
        raise ValueError(
            f"Unknown role {role!r}. Valid roles: {list(ROLE_SLOTS)}"
        )
    inv_perm = _ROLE_INV_PERMS[role]
    role_query = event_vec[inv_perm]
    return cosine(role_query, candidate)
