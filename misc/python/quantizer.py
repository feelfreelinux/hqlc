"""Scalar quantization: gain code, per-band step size, deadzone quantizer, noise fill.

Each MDCT coefficient becomes an integer symbol via a deadzone rule
  q = sign(x) * floor(|x|/step - DEAD_ZONE + 1)   if |x|/step > DEAD_ZONE, else 0
and is reconstructed with an MMSE-optimal centroid offset
  x_hat = sign(q) * (|q| + CENTROID) * step

step = 2^((exp_idx - 43) / 4) / gain folds the per-band log-energy exponent
(6-bit, from psy.py) with the global gain (7-bit code, 8 steps per octave).
Equivalent to the C reference step = 2^((2*exp - gain_code - 59) / 8).
"""

import math

import numpy as np

from .constants import (
    BAND_EDGES,
    CENTROID,
    DEAD_ZONE,
    EXP_INDEX_BIAS,
    N_BANDS,
)

# Gain quantization: 7-bit code, log2 Q3 (8 codes per octave)
GAIN_BITS = 7
GAIN_Q = 8
GAIN_BIAS = 27  # code 59 / gain 16 / 128 kbps-ish with rANS
GAIN_MAX_CODE = (1 << GAIN_BITS) - 1  # 127
GAIN_RC_MAX = GAIN_BIAS + GAIN_Q * 6  # cap: gain = 64 (6 octaves above unity)

# Noise fill seed, see quant.h
NF_SEED_BIAS = 0x9E3779B9
NF_SEED_CH = 0x51ED
NF_SEED_MUL = 0x01000193  # 32-bit FNV prime


def quantize_gain(gain):
    """Encode a gain value to a 7-bit code."""
    code = int(round(math.log2(max(1e-12, gain)) * GAIN_Q)) + GAIN_BIAS
    return max(0, min(GAIN_MAX_CODE, code))


def dequantize_gain(code):
    """Decode a 7-bit gain code to a gain value."""
    return 2.0 ** ((code - GAIN_BIAS) / float(GAIN_Q))


def exp_to_step_factor(exp_val):
    """Exponent index -> step scale factor 2^((exp - 43)/4) (scalar or array)."""
    return 2.0 ** ((exp_val - EXP_INDEX_BIAS) / 4.0)


def quantize(X, step):
    """Deadzone-quantize coefficients at a per-bin step size.

    Returns the integer symbols. step fixes the length. X is sliced to match (the inaudible HF tail is dropped).
    """
    n = len(step)
    scaled = np.zeros(n)
    valid = step > 1e-20  # guard against divide-by-zero on empty bands
    scaled[valid] = X[:n][valid] / step[valid]
    abs_scaled = np.abs(scaled)

    q = np.zeros(n, dtype=np.int32)
    mask = abs_scaled > DEAD_ZONE
    q[mask] = (
        np.sign(scaled[mask]) * np.floor(abs_scaled[mask] - DEAD_ZONE + 1.0)
    ).astype(np.int32)
    return q


def dequantize(q, step):
    """Centroid reconstruction: x_hat = sign(q) * (|q| + CENTROID) * step."""
    x = np.zeros(len(step))
    nz = q != 0
    x[nz] = np.sign(q[nz]) * (np.abs(q[nz]) + CENTROID) * step[nz]
    return x


# Fill amplitude at full occupancy, in quantizer steps. The C reference stores
# this rounded to Q8 (quant.c NF_LEVEL_Q8).
NF_LEVEL = 0.22

# First coarse band eligible for noise fill, ~2 kHz at low rates, ~3.5 kHz above
NF_FIRST_BAND = 10
NF_FIRST_BAND_LOW = 7

# A zero bin overshoots when the fill is more than this many times its true magnitude
NF_KILL_BIN_RATIO = 2

# Per-band NF mode, transmitted from NF_FIRST_BAND up
NF_FILL, NF_ATT, NF_OFF = 0, 1, 2

# From this gain code up killed bands are never filled and the fill starts higher
NF_ATT_GAIN_MAX = 49


def nf_first_band(gain_code):
    return NF_FIRST_BAND_LOW if gain_code < NF_ATT_GAIN_MAX else NF_FIRST_BAND

# A killed band keeps an attenuated fill (-9 dB) when at least half of its zero bins
# sit within this ratio of the estimate (roughly 15 dB)
NF_ATT_BIN_RATIO = 6
NF_ATT_LEVEL = 0.355


def _nf_band_factor(nz, width):
    """Fill level of a band in quantizer steps, from its zero count.

    Only the excess of zeros over nonzeros counts as evidence of a missing noise floor,
    so a band that is less than half zeros is not filled (returns 0)
    """
    excess = 2.0 * nz / float(width) - 1.0
    return NF_LEVEL * excess if excess > 0.0 else 0.0


def nf_kill_mask(X, q, step, gain_code):
    """Encoder-side NF kill mask: bands where the decoder's fill would overshoot.

    Mirrors the decoder level estimate, a band is killed when more than half of its
    zero bins would be filled above NF_KILL_BIN_RATIO x their true magnitude. Below
    NF_ATT_GAIN_MAX a killed band keeps an attenuated fill when at least half of its
    zero bins are within NF_ATT_BIN_RATIO of the estimate.
    """
    mask = np.zeros(N_BANDS, dtype=np.uint8)
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        if b < nf_first_band(gain_code):
            continue
        zeros = [i for i in range(s, e) if q[i] == 0]
        factor = _nf_band_factor(len(zeros), e - s)
        if factor == 0.0:
            continue
        overshoot = sum(1 for i in zeros if factor > NF_KILL_BIN_RATIO * abs(X[i]) / step[i])
        if 2 * overshoot <= len(zeros):
            continue
        mask[b] = NF_OFF
        if gain_code < NF_ATT_GAIN_MAX:
            near = sum(1 for i in zeros if NF_ATT_BIN_RATIO * abs(X[i]) / step[i] > factor)
            if 2 * near >= len(zeros):
                mask[b] = NF_ATT
    return mask


def noise_fill(X_hat, q, step, seed, gain_code, band_mode=None):
    """Replace zeroed bins from nf_first_band(gain_code) up with pseudorandom noise, in place.

    The fill level comes from each band's zero occupancy (see _nf_band_factor).

    band_mode (per band) leaves NF_OFF bands as exact zeros, used for the M/S side
    bands and the NF kill mask, and attenuates NF_ATT bands
    """
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        if b < nf_first_band(gain_code) or (band_mode is not None and band_mode[b] == NF_OFF):
            continue

        zeros = [i for i in range(s, e) if q[i] == 0]
        factor = _nf_band_factor(len(zeros), e - s)
        if band_mode is not None and band_mode[b] == NF_ATT:
            factor *= NF_ATT_LEVEL
        if factor == 0.0:
            continue

        for i in zeros:
            # xorshift
            seed ^= (seed << 13) & 0xFFFFFFFF
            seed ^= seed >> 17
            seed ^= (seed << 5) & 0xFFFFFFFF

            sign = -1.0 if (seed & 0x80000000) else 1.0
            X_hat[i] = sign * factor * step[i]

    return seed
