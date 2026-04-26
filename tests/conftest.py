"""
Shared fixtures for the MDA test suite.
All tests import from the repo root, so sys.path is patched here.
"""
import sys
import os

# Ensure the repo root is on the path so "from mda.xxx import ..." works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from mda.core.bind import DIM, random_vector
from mda.core.encoder import HolisticEncoder
from mda.core.registry import EntityRegistry
from mda.core.entity import Entity
from mda.inference.broca import BrocaModule


@pytest.fixture
def dim():
    return DIM


@pytest.fixture
def vec():
    """A reproducible random unit vector."""
    return random_vector(DIM, seed=0)


@pytest.fixture
def encoder():
    return HolisticEncoder(DIM)


@pytest.fixture
def registry():
    return EntityRegistry()


@pytest.fixture
def broca(encoder, registry):
    return BrocaModule(encoder, registry)


@pytest.fixture
def entity():
    """A minimal Entity with a known surface."""
    return Entity(id="e_000001", surface="TestEntity")


@pytest.fixture
def entity_pair(registry):
    """Two related entities already in the registry."""
    a = registry.get_or_create("Alpha", "concept")
    b = registry.get_or_create("Beta",  "concept")
    return a, b
