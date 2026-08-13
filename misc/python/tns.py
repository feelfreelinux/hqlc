"""Temporal Noise Shaping (TNS)

TNS applies linear prediction in the frequency domain to shape quantization
noise in the time domain, hiding it behind transients instead of smearing / pre-echo

Only bins >= TNS_START_BIN (~940 Hz) are analysed and filtered.
"""

import numpy as np

from .constants import FRAME_LEN

TNS_MAX_ORDER = 4
TNS_MAX_K = 0.92
TNS_K_BITS = 4
TNS_LAR_MAX = 3.5
TNS_START_BIN = 20  # ~940 Hz

# Hangover frames use a softer window, and a stricter gate
TNS_LAG_WIN_B = 0.03
TNS_LAG_WIN_B_HANGOVER = 0.05

TNS_PRED_GAIN_THR = 1.2
TNS_PRED_GAIN_THR_HANGOVER = 1.5


# Sub-block energy below this is treated as silence (the C impl carries it in Q30)
TNS_DETECT_FLOOR = 2.0**-10

TNS_DETECT_SUBBLOCKS = 8
TNS_DETECT_RATIO = 8


class TransientDetector:
    """Simple time-domain attack detector:
    high-passed signal is split into 8 sub blocks, each compared against the mean of the preceding 8 sub-blocks.

    Stateful so we work across the frame boundary"""

    def __init__(self):
        self.sub_energy = [0.0] * TNS_DETECT_SUBBLOCKS
        self.last = 0.0

    def detect(self, frame):
        sub = FRAME_LEN // TNS_DETECT_SUBBLOCKS
        e = []
        last = self.last
        for b in range(TNS_DETECT_SUBBLOCKS):
            blk = frame[b * sub : (b + 1) * sub]

            # First-difference preemphasis: y[n] = x[n] - x[n-1]
            # Done so we don't let the LF energy dominate the TNS detection,
            # TODO: Check if this is the right filter setup, maybe we should have a proper cutoff frequency filter here
            d = np.diff(np.concatenate(([last], blk)))
            e.append(float(np.dot(d, d)))
            last = float(blk[-1])
        self.last = last

        win = sum(self.sub_energy)
        fire = False
        for b in range(TNS_DETECT_SUBBLOCKS):
            if (
                e[b] > TNS_DETECT_FLOOR
                and e[b] * TNS_DETECT_SUBBLOCKS > TNS_DETECT_RATIO * win
            ):
                fire = True
            win += e[b] - self.sub_energy[b]
        self.sub_energy = e
        return fire


def _autocorrelation(x, max_order):
    """Autocorrelation r[k] = sum x[n]*x[n+k] for k = 0..max_order."""
    n = len(x)
    r = np.empty(max_order + 1, dtype=np.float64)
    for k in range(max_order + 1):
        r[k] = np.dot(x[: n - k], x[k:n]) if k < n else 0.0
    return r


def _levinson_durbin(r, max_order):
    """Does the levinson-durbin recursion to compute (reflection_coeffs, order, prediction_gain)"""
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
    # f = forward / whitened path, b_state = one backward delay per stage
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
    # Undo the FIR: recover f stage by stage, then refill the backward delays
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


def analyze(X, hangover=False):
    """TNS analysis on the HF spectrum (bins >= TNS_START_BIN).

    Needs to be transient-gated at the caller (detector + hangover)

    Returns (order, k_dequantized, q_indices, side_bits).
    """
    r = _autocorrelation(X[TNS_START_BIN:], TNS_MAX_ORDER)
    # Gaussian lag window, mirrors the c codec
    b = TNS_LAG_WIN_B_HANGOVER if hangover else TNS_LAG_WIN_B
    lag = np.arange(1, TNS_MAX_ORDER + 1)
    r[1:] = r[1:] * np.exp(-0.5 * (2.0 * np.pi * b * lag) ** 2)
    k_raw, order, pred_gain = _levinson_durbin(r, TNS_MAX_ORDER)

    thr = TNS_PRED_GAIN_THR_HANGOVER if hangover else TNS_PRED_GAIN_THR
    if order == 0 or pred_gain < thr:
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
