"""MDCT analysis and synthesis using a KBD window and DCT-IV.

The Modified Discrete Cosine Transform is a lapped transform:
  X[k] = sum_{n=0}^{2N-1} w[n] * x[n] * cos(pi/N * (n + 0.5 + N/2) * (k + 0.5))

Implemented as: window -> TDAC fold -> DCT-IV (analysis)
                DCT-IV -> unfold -> window (synthesis)

The DCT-IV is self-inverse: dct4(dct4(x)) = (N/2) * x
"""

import numpy as np

KBD_ALPHA = 3.0

_window_cache: dict[int, np.ndarray] = {}
_cos_dct4_cache: dict[int, np.ndarray] = {}


def make_kbd_window(N: int, alpha: float = KBD_ALPHA) -> np.ndarray:
    """Kaiser-Bessel Derived window of length 2N."""
    beta = np.pi * alpha
    w_k = np.i0(
        beta * np.sqrt(1.0 - ((2.0 * np.arange(N + 1) / N) - 1.0) ** 2)
    ) / np.i0(beta)
    W = np.cumsum(w_k)
    kbd = np.zeros(2 * N)
    kbd[:N] = np.sqrt(W[:N] / W[N])
    kbd[N:] = kbd[N - 1 :: -1]
    return kbd


def get_window(N: int) -> np.ndarray:
    """Return cached KBD window of length 2N."""
    win = _window_cache.get(N)
    if win is None:
        win = make_kbd_window(N)
        _window_cache[N] = win
    return win


def _dct4_cos_matrix(N: int) -> np.ndarray:
    """Cosine basis for DCT-IV: C[n,k] = cos(pi/N * (n+0.5) * (k+0.5))."""
    C = _cos_dct4_cache.get(N)
    if C is None:
        n = np.arange(N, dtype=np.float64)
        k = np.arange(N, dtype=np.float64)
        C = np.cos((np.pi / float(N)) * np.outer(n + 0.5, k + 0.5))
        _cos_dct4_cache[N] = C
    return C


def dct4(x: np.ndarray) -> np.ndarray:
    """Unnormalized DCT-IV. Self-inverse: dct4(dct4(x)) = (N/2) * x."""
    return np.dot(np.asarray(x, dtype=np.float64), _dct4_cos_matrix(x.size))


def idct4(X: np.ndarray) -> np.ndarray:
    """Inverse DCT-IV: x = (2/N) * dct4(X)."""
    return (2.0 / float(X.size)) * dct4(X)


def _fold(xw: np.ndarray) -> np.ndarray:
    """TDAC fold, windowed 2N samples -> N spectral input samples."""
    N = xw.size // 2
    N2 = N // 2
    folded = np.empty(N, dtype=np.float64)
    n = np.arange(N2)
    folded[:N2] = -(xw[3 * N2 + n] + xw[3 * N2 - 1 - n])
    folded[N2:] = xw[n] - xw[N - 1 - n]
    return folded


def _unfold(u: np.ndarray) -> np.ndarray:
    """Inverse TDAC fold, spectral output -> 2N pre-windowed samples."""
    N = u.size
    N2 = N // 2
    out = np.empty(2 * N, dtype=np.float64)
    n = np.arange(N2)
    out[:N2] = u[N2 + n]
    out[N2:N] = -u[N - 1 - n]
    out[N : N + N2] = -u[N2 - 1 - n]
    out[N + N2 :] = -u[n]
    return out


def mdct_analysis(x: np.ndarray) -> np.ndarray:
    """MDCT analysis, 2N time samples -> N spectral coefficients."""
    N = x.size // 2
    return dct4(_fold(x * get_window(N)))


def imdct_synthesis(X: np.ndarray) -> np.ndarray:
    """Inverse MDCT, N spectral coefficients -> 2N windowed time samples."""
    N = X.size
    return _unfold(idct4(X)) * get_window(N)
