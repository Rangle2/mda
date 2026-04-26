import numpy as np

DIM = 256


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-8)


def random_vector(dim: int = DIM, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(0, 1, dim))


def zero_vector(dim: int = DIM) -> np.ndarray:
    return np.zeros(dim)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
