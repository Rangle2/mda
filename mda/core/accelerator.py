"""
mda/core/accelerator.py — numpy/torch adapter.

All GPU work happens here; everything else in MDA stays numpy-typed at call
boundaries.  Graceful-fallback pattern: torch is an optional dependency.
HAS_TORCH / HAS_CUDA / HAS_MPS flags are set at import time and can be
inspected by callers before dispatching to GPU helpers.

Functions that unconditionally require torch (to_t, to_np, normalize_t,
cosine_t, fft_bind_t) raise RuntimeError when torch is absent rather than
producing a confusing NameError.  Batch helpers (batch_cosine, batch_matmul)
fall back to numpy silently.
"""
import numpy as np

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA  = torch.cuda.is_available()
    HAS_MPS   = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )

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
    """Return the active torch.device, or None if torch is not installed."""
    return DEVICE


def to_t(v: np.ndarray):
    """np float32 array → contiguous torch.Tensor on DEVICE.

    Raises RuntimeError if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("to_t requires torch, which is not installed.")
    return torch.from_numpy(np.ascontiguousarray(v)).float().to(DEVICE)


def to_np(t) -> np.ndarray:
    """torch.Tensor (any device) → np.float32 ndarray.

    Raises RuntimeError if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("to_np requires torch, which is not installed.")
    return t.detach().cpu().numpy().astype(np.float32)


def normalize_t(v: np.ndarray) -> np.ndarray:
    """Normalize *v* via torch.linalg.vector_norm on DEVICE; return np.float32.

    Raises RuntimeError if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("normalize_t requires torch, which is not installed.")
    t = to_t(v)
    n = torch.linalg.vector_norm(t)
    return to_np(t / (n + 1e-8))


def cosine_t(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity via torch.dot on DEVICE; return Python float.

    Raises RuntimeError if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("cosine_t requires torch, which is not installed.")
    ta = to_t(a)
    tb = to_t(b)
    na = torch.linalg.vector_norm(ta)
    nb = torch.linalg.vector_norm(tb)
    return float((torch.dot(ta, tb) / (na * nb + 1e-8)).item())


def fft_bind_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution via torch.fft on DEVICE; return real np.float32.

    Raises RuntimeError if torch is not installed.
    """
    if not HAS_TORCH:
        raise RuntimeError("fft_bind_t requires torch, which is not installed.")
    ta = to_t(a)
    tb = to_t(b)
    result = torch.fft.ifft(torch.fft.fft(ta) * torch.fft.fft(tb))
    return to_np(result.real)


def batch_cosine(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """(N, D) matrix @ (D,) vec → (N,) cosine scores as np.float32.

    Matrix rows are assumed to be pre-normalised (unit vectors).
    Uses torch.mv on DEVICE when torch is available; falls back to numpy.
    """
    if HAS_TORCH:
        tm = to_t(matrix)
        tv = to_t(vec)
        return torch.mv(tm, tv).cpu().numpy()
    return (matrix @ vec).astype(np.float32)


def batch_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """General (M, K) @ (K, N) → (M, N) as np.float32.

    Uses torch.mm on DEVICE when torch is available; falls back to numpy.
    """
    if HAS_TORCH:
        ta = to_t(a)
        tb = to_t(b)
        return torch.mm(ta, tb).cpu().numpy()
    return (a @ b).astype(np.float32)


# ---------------------------------------------------------------------------
# MDA Dual Mode — SINGLE (numpy/CPU) vs BATCH (CUDA, high throughput)
# ---------------------------------------------------------------------------

class MDAMode:
    SINGLE = "single"
    BATCH  = "batch"


_mode: str = MDAMode.SINGLE


def set_mode(mode: str) -> None:
    """Switch between MDAMode.SINGLE and MDAMode.BATCH."""
    global _mode
    _mode = mode


def get_mode() -> str:
    """Return the current mode string ('single' or 'batch')."""
    return _mode


def is_batch_mode() -> bool:
    """True only when mode is BATCH *and* CUDA is available."""
    return _mode == MDAMode.BATCH and HAS_CUDA
