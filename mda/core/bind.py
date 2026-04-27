import numpy as np

DIM = 512

# GPU delegation — safe regardless of torch availability.
try:
    from mda.core.accelerator import (
        normalize_t as _normalize_t,
        cosine_t    as _cosine_t,
        fft_bind_t  as _fft_bind_t,
        HAS_TORCH,
    )
except ImportError:
    HAS_TORCH    = False
    _normalize_t = None
    _cosine_t    = None
    _fft_bind_t  = None


def normalize(v: np.ndarray) -> np.ndarray:
    if HAS_TORCH and _normalize_t is not None:
        return _normalize_t(v)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def random_vector(dim: int = DIM, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(0, 1, dim))


def zero_vector(dim: int = DIM) -> np.ndarray:
    return np.zeros(dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if HAS_TORCH and _fft_bind_t is not None:
        return _fft_bind_t(a, b)
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def unbind(compound: np.ndarray, b: np.ndarray) -> np.ndarray:
    F_b     = np.fft.fft(b)
    F_b_inv = np.conj(F_b) / (np.abs(F_b) ** 2 + 1e-6)
    b_inv   = np.real(np.fft.ifft(F_b_inv))
    return np.real(np.fft.ifft(np.fft.fft(compound) * np.fft.fft(b_inv)))


def bind_many(*vectors: np.ndarray) -> np.ndarray:
    result = vectors[0].copy()
    for v in vectors[1:]:
        result = bind(result, v)
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if HAS_TORCH and _cosine_t is not None:
        return _cosine_t(a, b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
