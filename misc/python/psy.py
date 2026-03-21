"""Psychoacoustic analysis: band energy, spreading envelope, noise fill decisions.

Band energy is computed as a log-scale exponent index per band:
  exp_idx = clip(round(2 * log2(mean_sq)) + BIAS, 0, 63)
This gives ~1.5 dB granularity and controls the quantizer step size.

The spreading envelope models cross-band masking using the ERB-rate scale
(Glasberg & Moore 1990). Upward masking decays at -10 dB/ERB, downward
at -15 dB/ERB. The signal-to-mask ratio (SMR) drives tier-2 noise fill.

Noise fill replaces very quiet or deeply masked bands with shaped pseudorandom
noise, avoiding wasteful coding of near-zero coefficients.
  - Tier 1: low energy (exp_idx <= 7) + non-tonal (crest factor < 20 dB)
  - Tier 2: deeply masked (exp_idx <= 24, SMR below threshold)
"""

import math

import numpy as np

from .constants import (
    BAND_EDGES, BLOCK_SIZE, EXP_INDEX_BIAS, EXP_INDEX_MAX, EXP_INDEX_MIN,
    FS, LOG2_BINS_Q4, N_BANDS,
)

# Noise fill parameters
NF_EXP_MAX = 7
NF_MUL = 0.7
NF_CREST_DB = 20.0
NF_EXP_MAX_TIER2 = 24
NF_SMR_THRESHOLD_Q4 = 8  # Q4(0.5)

_NF_CREST_RATIO = 10.0 ** (NF_CREST_DB / 10.0)  # 100.0

# Noise fill LCG constants (Numerical Recipes)
NF_LCG_A = 1664525
NF_LCG_C = 1013904223
NF_SEED_BIAS = 0x9E3779B9


# ── Spreading envelope ──

def _erb_rate(hz):
    """ERB-rate scale (Glasberg & Moore 1990)."""
    return 21.4 * math.log10(1 + 0.00437 * hz)


_HZ_PER_BIN = FS / BLOCK_SIZE
_ERB_CENTERS = np.array([
    _erb_rate((BAND_EDGES[b] + BAND_EDGES[b + 1]) * _HZ_PER_BIN / 2.0)
    for b in range(N_BANDS)
])

_SPREAD_UP_DB_PER_ERB = -10.0
_SPREAD_DOWN_DB_PER_ERB = -15.0
_DB_TO_Q4 = 16.0 / 3.0103  # dB -> Q4 log2 (power domain)
_Q4_NEG_INF = -10000

_SPREAD_DECAY_UP_Q4 = np.zeros(N_BANDS, dtype=np.int32)
_SPREAD_DECAY_DOWN_Q4 = np.zeros(N_BANDS, dtype=np.int32)
for _b in range(1, N_BANDS):
    _erb_c2c = _ERB_CENTERS[_b] - _ERB_CENTERS[_b - 1]
    _SPREAD_DECAY_UP_Q4[_b] = int(round(_SPREAD_UP_DB_PER_ERB * _erb_c2c * _DB_TO_Q4))
    _SPREAD_DECAY_DOWN_Q4[_b - 1] = int(
        round(_SPREAD_DOWN_DB_PER_ERB * _erb_c2c * _DB_TO_Q4)
    )


def compute_spreading_envelope(exp_indices):
    """Cross-band masking envelope. Returns (energy_q4, mask_q4)."""
    energy_q4 = np.array(
        [(int(exp_indices[b]) - EXP_INDEX_BIAS) * 8 for b in range(N_BANDS)],
        dtype=np.int32,
    )
    total_q4 = energy_q4 + LOG2_BINS_Q4

    # Forward sweep (upward masking)
    mask_left = np.full(N_BANDS, _Q4_NEG_INF, dtype=np.int32)
    for b in range(1, N_BANDS):
        mask_left[b] = max(total_q4[b - 1], mask_left[b - 1]) + _SPREAD_DECAY_UP_Q4[b]

    # Backward sweep (downward masking)
    mask_right = np.full(N_BANDS, _Q4_NEG_INF, dtype=np.int32)
    for b in range(N_BANDS - 2, -1, -1):
        mask_right[b] = max(total_q4[b + 1], mask_right[b + 1]) + _SPREAD_DECAY_DOWN_Q4[b]

    mask_q4 = np.maximum(mask_left, mask_right) - LOG2_BINS_Q4
    return energy_q4, mask_q4


def compute_exponents(X):
    """Compute band exponent indices from MDCT coefficients (one channel)."""
    exp = np.zeros(N_BANDS, dtype=np.int32)
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        w = e - s
        mean_sq = np.sum(X[s:e] ** 2) / float(w) if w > 0 else 0.0
        if mean_sq > 1e-18:
            exp[b] = int(np.clip(
                round(2.0 * math.log2(mean_sq)) + EXP_INDEX_BIAS,
                EXP_INDEX_MIN, EXP_INDEX_MAX,
            ))
    return exp


def nf_crest_below(band):
    """Test if crest factor is below NF_CREST_DB (non-tonal)."""
    w = len(band)
    if w == 0:
        return True
    peak = float(np.max(np.abs(band)))
    sum_sq = float(np.sum(band * band))
    if sum_sq < 1e-30:
        return True
    return peak * peak * w < _NF_CREST_RATIO * sum_sq


def nf_decide(exp_indices_ch, Xs_ch, smr_q4_ch=None):
    """Two-tier noise fill decision per band."""
    nf = np.zeros(N_BANDS, dtype=bool)
    for b in range(N_BANDS):
        idx = int(exp_indices_ch[b])
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        if idx <= NF_EXP_MAX and nf_crest_below(Xs_ch[s:e]):
            nf[b] = True
        elif smr_q4_ch is not None and idx <= NF_EXP_MAX_TIER2:
            if smr_q4_ch[b] <= NF_SMR_THRESHOLD_Q4:
                nf[b] = True
    return nf


def nf_synthesize(seed, n, amp):
    """LCG noise with L1 normalization at gain-invariant amplitude."""
    noise = np.zeros(n, dtype=np.float64)
    x = int(seed)
    mean = 0.0
    for i in range(n):
        x = (NF_LCG_A * x + NF_LCG_C) & 0xFFFFFFFF
        u = float((x >> 8) & 0xFFFFFF) * (1.0 / 16777216.0)
        v = 2.0 * u - 1.0
        noise[i] = v
        mean += v
    if n > 0:
        noise -= mean / n
        mean_abs = float(np.mean(np.abs(noise)))
        if mean_abs > 1e-12 and amp != 0.0:
            noise *= amp / (1.1547005383792517 * mean_abs)  # 2/sqrt(3)
            return noise
    return noise * amp
