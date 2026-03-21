"""Temporal Noise Shaping (TNS).

TNS applies linear prediction in the frequency domain to shape quantization
noise in the time domain. This concentrates noise where the signal is loud,
reducing audible pre-echo artifacts on transient signals.

TNS operates only on high-frequency bins (above TNS_START_BIN, ~2 kHz).
This prevents the filter from distorting low-frequency tonal content
(e.g., bass instruments) while still providing full pre-echo suppression
in the perceptually sensitive high-frequency range.

Algorithm:
  1. Detect transients via energy ratio test (threshold 2x) on the time-domain block.
  2. Compute autocorrelation of the HF portion of the MDCT spectrum.
  3. Solve for reflection coefficients via Levinson-Durbin.
  4. Quantize reflection coefficients through the LAR (Log Area Ratio) domain.
  5. Apply a lattice FIR filter to the HF spectrum (encoder).
  6. Apply the inverse lattice IIR filter at the decoder.
"""

import numpy as np

from .constants import FRAME_LEN

TNS_MAX_ORDER = 4
TNS_MAX_K = 0.85
TNS_PRED_GAIN_THR = 1.5
TNS_ATTACK_RATIO = 2.0
TNS_K_BITS = 4
TNS_LAR_MAX = 3.5
TNS_START_BIN = 43  # ~2 kHz: analyse and filter only above this bin


def _autocorrelation(x, max_order):
    """Autocorrelation r[k] = sum x[n]*x[n+k] for k = 0..max_order."""
    n = len(x)
    r = np.empty(max_order + 1, dtype=np.float64)
    for k in range(max_order + 1):
        r[k] = np.dot(x[: n - k], x[k:n]) if k < n else 0.0
    return r


def _levinson_durbin(r, max_order, k_threshold=0.1):
    """Levinson-Durbin recursion -> (reflection_coeffs, order, prediction_gain)."""
    if r[0] < 1e-30:
        return np.zeros(0, dtype=np.float64), 0, 1.0
    error = r[0]
    a = np.zeros(max_order, dtype=np.float64)
    k_out = []
    for i in range(max_order):
        acc = r[i + 1]
        for j in range(i):
            acc += a[j] * r[i - j]
        ki = np.clip(-acc / error, -0.999, 0.999)
        if abs(ki) < k_threshold:
            break
        error *= 1.0 - ki * ki
        if error < 1e-30:
            break
        k_out.append(ki)
        new_a = np.zeros(max_order, dtype=np.float64)
        for j in range(i):
            new_a[j] = a[j] + ki * a[i - 1 - j]
        new_a[i] = ki
        a = new_a
    order = len(k_out)
    pred_gain = r[0] / error if error > 1e-30 and order > 0 else 1.0
    return np.array(k_out, dtype=np.float64), order, pred_gain


def _quant_lar(k_raw):
    """Quantize reflection coefficients via LAR domain (4-bit, range +/-7)."""
    half = (1 << (TNS_K_BITS - 1)) - 1  # 7
    k_clipped = np.clip(k_raw, -0.999, 0.999)
    lar = np.log((1.0 + k_clipped) / (1.0 - k_clipped))
    q = np.clip(np.rint(lar * half / TNS_LAR_MAX).astype(np.int32), -half, half)
    k_dq = np.tanh(q.astype(np.float64) * TNS_LAR_MAX / float(half) / 2.0)
    return q, k_dq


def lattice_fir(x, k):
    """Lattice FIR analysis filter (forward, applied at encoder).

    Processes only bins >= TNS_START_BIN, leaving LF untouched.
    """
    M = len(k)
    if M == 0:
        return x.copy()
    N = len(x)
    y = x.copy()
    b_state = np.zeros(M, dtype=np.float64)
    for n in range(TNS_START_BIN, N):
        f = x[n]
        b_new = np.empty(M, dtype=np.float64)
        b_new[0] = x[n]
        for i in range(M):
            f_next = f + k[i] * b_state[i]
            if i + 1 < M:
                b_new[i + 1] = k[i] * f + b_state[i]
            f = f_next
        y[n] = f
        b_state = b_new
    return y


def lattice_iir(y, k):
    """Lattice IIR synthesis filter (inverse of FIR, applied at decoder).

    Processes only bins >= TNS_START_BIN, leaving LF untouched.
    """
    M = len(k)
    if M == 0:
        return y.copy()
    N = len(y)
    x = y.copy()
    b_state = np.zeros(M, dtype=np.float64)
    for n in range(TNS_START_BIN, N):
        f = y[n]
        b_new = np.empty(M, dtype=np.float64)
        for i in range(M - 1, -1, -1):
            f = f - k[i] * b_state[i]
            if i + 1 < M:
                b_new[i + 1] = k[i] * f + b_state[i]
        x[n] = f
        b_new[0] = f
        b_state = b_new
    return x


def analyze(X, block):
    """TNS analysis with transient gating.

    Analyses only the HF portion of the spectrum (bins >= TNS_START_BIN).
    Returns (order, k_dequantized, q_indices, side_bits).
    """
    N = FRAME_LEN

    # Transient detection: compare energy of first and second halves
    if TNS_ATTACK_RATIO > 0:
        h1, h2 = block[:N], block[N:]
        e1 = float(np.dot(h1, h1)) + 1e-30
        e2 = float(np.dot(h2, h2))
        if (e2 / e1) < TNS_ATTACK_RATIO:
            return 0, np.zeros(0), np.zeros(0, dtype=np.int32), 1

    # Autocorrelation on HF bins only
    r = _autocorrelation(X[TNS_START_BIN:], TNS_MAX_ORDER)
    k_raw, order, pred_gain = _levinson_durbin(r, TNS_MAX_ORDER)

    if order == 0 or pred_gain < TNS_PRED_GAIN_THR:
        return 0, np.zeros(0), np.zeros(0, dtype=np.int32), 1

    k_raw = np.clip(k_raw, -TNS_MAX_K, TNS_MAX_K)
    q_k, k_dq = _quant_lar(k_raw)

    # Trim trailing zeros
    while len(k_dq) > 0 and q_k[-1] == 0:
        q_k, k_dq = q_k[:-1], k_dq[:-1]
    if len(k_dq) == 0:
        return 0, np.zeros(0), np.zeros(0, dtype=np.int32), 1

    order = len(k_dq)
    side_bits = 1 + 3 + TNS_K_BITS * order
    return order, k_dq, q_k, side_bits
