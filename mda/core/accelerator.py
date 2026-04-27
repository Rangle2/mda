"""
mda/core/accelerator.py — numpy/torch adapter.
All GPU work happens here. Everything else in MDA stays numpy-typed at call boundaries.
"""
import numpy as np

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA  = torch.cuda.is_available()
    HAS_MPS   = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if HAS_CUDA:
        DEVICE = torch.device("cuda")
    elif HAS_MPS:
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

except ImportError:
    HAS_TORCH = False
    HAS_CUDA  = False
    HAS_MPS   = False
    DEVICE    = None


def get_device():
    return DEVICE


def to_t(v: np.ndarray):
    """np float32 array → tensor on DEVICE."""
    return torch.from_numpy(np.ascontiguousarray(v)).float().to(DEVICE)


def to_np(t) -> np.ndarray:
    """tensor (any device) → np float32 array."""
    return t.detach().cpu().numpy().astype(np.float32)


def normalize_t(v: np.ndarray) -> np.ndarray:
    """torch linalg.vector_norm path, returns np."""
    t = to_t(v)
    n = torch.linalg.vector_norm(t)
    return to_np(t / (n + 1e-8))


def cosine_t(a: np.ndarray, b: np.ndarray) -> float:
    """torch.dot path, returns float."""
    ta = to_t(a)
    tb = to_t(b)
    na = torch.linalg.vector_norm(ta)
    nb = torch.linalg.vector_norm(tb)
    return float((torch.dot(ta, tb) / (na * nb + 1e-8)).item())


def fft_bind_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """torch.fft.fft/ifft bind, returns np. Real part only."""
    ta = to_t(a)
    tb = to_t(b)
    result = torch.fft.ifft(torch.fft.fft(ta) * torch.fft.fft(tb))
    return to_np(result.real)


def batch_cosine(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    (N, D) matrix @ (D,) vec → (N,) cosine scores.
    matrix rows assumed already normalized.
    Falls back to np.dot if no torch.
    """
    if HAS_TORCH:
        tm = to_t(matrix)
        tv = to_t(vec)
        return torch.mv(tm, tv).cpu().numpy()
    return matrix @ vec


def batch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """General (M, K) @ (K, N) → (M, N)."""
    if HAS_TORCH:
        ta = to_t(a)
        tb = to_t(b)
        return torch.mm(ta, tb).cpu().numpy()
    return a @ b
