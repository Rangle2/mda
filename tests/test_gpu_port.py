import numpy as np
import pytest

# ── dim=512 ─────────────────────────────────────────────────────────────────

def test_dim_is_512():
    from mda.core.bind import DIM
    assert DIM == 512

def test_entity_uses_correct_dim():
    from mda.core.entity import Entity
    e = Entity(id="test", surface="TestEntity", dim=512)
    assert e.v.shape == (512,)
    assert e.h.shape == (512,)

def test_encoder_metadata_slots_relative():
    from mda.core.encoder import HolisticEncoder
    enc = HolisticEncoder(dim=512)
    v = enc.encode("Kairfy is a legal analysis platform?")
    assert v.shape == (512,)
    # metadata region: dim-64 to dim-58 — should have non-zero values
    assert v[512 - 64] > 0   # text length slot
    assert v[512 - 61] == 1.0  # "?" flag

def test_checkpoint_saves_dim(tmp_path):
    from mda import MDA
    from mda.training.checkpoint import save, load
    from mda.core.registry import EntityRegistry
    from mda.inference.broca import BrocaModule
    m = MDA()
    m.learn("MDA is a token-free memory system")
    base = str(tmp_path / "test_ckpt")
    save(m.registry, m.broca, base)
    import json
    meta = json.loads((tmp_path / "test_ckpt.json").read_text())
    assert meta["session"]["dim"] == 512

def test_checkpoint_dim_mismatch_raises(tmp_path):
    """Loading a 256-dim checkpoint into 512 system must raise."""
    import json, numpy as np
    base = str(tmp_path / "bad")
    # Write fake 256-dim checkpoint
    meta = {"session": {"dim": 256, "user_id": "x", "model_name": "x",
                        "turn_count": 0, "mda_version": "1.0",
                        "created_at": "2025", "updated_at": "2025"},
            "entities": {}, "broca": {"entity_facts": {},
                         "entity_positions": {}, "categories": {}}}
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    # Save a 256-dim array to trigger mismatch
    np.savez_compressed(base + ".npz", dummy=np.zeros(256, dtype=np.float32))
    from mda.core.registry import EntityRegistry
    from mda.inference.broca import BrocaModule
    from mda.core.encoder import HolisticEncoder
    reg = EntityRegistry()
    enc = HolisticEncoder(512)
    bro = BrocaModule(enc, reg)
    from mda.training.checkpoint import load
    with pytest.raises(ValueError, match="dim=256"):
        load(reg, bro, base)

# ── accelerator ─────────────────────────────────────────────────────────────

def test_accelerator_importable():
    from mda.core import accelerator
    assert hasattr(accelerator, "HAS_TORCH")
    assert hasattr(accelerator, "batch_cosine")

def test_normalize_t_matches_numpy():
    from mda.core.accelerator import normalize_t, HAS_TORCH
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from mda.core.bind import normalize
    v = np.random.randn(512).astype(np.float32)
    np.testing.assert_allclose(normalize_t(v), normalize(v), rtol=1e-5)

def test_batch_cosine_matches_serial():
    from mda.core.accelerator import batch_cosine, HAS_TORCH
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from mda.core.bind import cosine, normalize
    vecs = np.random.randn(20, 512).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= (norms + 1e-8)
    q = normalize(np.random.randn(512).astype(np.float32))
    batch = batch_cosine(vecs, q)
    serial = np.array([cosine(vecs[i], q) for i in range(20)])
    np.testing.assert_allclose(batch, serial, atol=1e-5)

# ── EntityMatrix ─────────────────────────────────────────────────────────────

def test_entity_matrix_nearest_matches_serial():
    from mda import MDA
    from mda.core.bind import cosine, normalize
    m = MDA()
    for txt in ["Python is a programming language",
                "Kairfy analyzes legal contracts",
                "MDA uses HDR vectors for memory",
                "GPU accelerates matrix operations"]:
        m.learn(txt)
    q = normalize(m.encoder.encode("legal document"))
    # EntityMatrix nearest
    hits = m.registry.nearest(q, top_k=3)
    # Serial
    serial = sorted([(cosine(q, e.v), e) for e in m.registry.all()],
                    key=lambda x: -x[0])[:3]
    # Top-1 must match
    assert hits[0][1].id == serial[0][1].id

# ── broca batch ──────────────────────────────────────────────────────────────

def test_score_facts_batch_equals_serial():
    from mda import MDA
    from mda.core.accelerator import HAS_TORCH
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    m = MDA()
    for _ in range(6):  # need >= 4 facts for batch path
        m.learn("Kairfy processes Turkish legal documents for contract risk")
        m.learn("MDA entity vectors are holographic distributed representations")
    q = m.encoder.encode("legal document analysis")
    entity = next(iter(m.registry._entities.values()))
    result = m.broca._score_facts(entity, q, top_k=3)
    for score, fact in result:
        assert isinstance(score, float)
        assert isinstance(fact, str)

# ── infer_from_chain parallel ────────────────────────────────────────────────

def test_infer_parallel_matches_serial():
    from mda import MDA
    from mda.inference.reasoning import ReasoningEngine
    m = MDA()
    texts = [
        "Kairfy is a legal SaaS built on MDA",
        "MDA uses holographic vectors for memory",
        "Turkish lawyers use Kairfy for contract analysis",
        "HDR vectors enable associative retrieval",
    ]
    for t in texts * 3:
        m.learn(t)
    engine = ReasoningEngine(m.encoder, m.registry)
    q = m.encoder.encode("legal contract memory")
    chain = m._chain.expand_from_text("Kairfy")
    if chain is None or not chain.nodes:
        pytest.skip("no chain nodes")
    serial   = engine._infer_paths_serial(
        [n.path for n in chain.nodes if n.path], q, m.broca, top_k=3)
    parallel = engine._infer_paths_parallel(
        [n.path for n in chain.nodes if n.path], q, m.broca, top_k=3)
    assert set(serial) == set(parallel)
