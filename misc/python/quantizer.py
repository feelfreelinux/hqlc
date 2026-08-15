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
    EXP_INDEX_MIN,
    N_BANDS,
)

# Gain quantization: 7-bit code, log2 Q3 (8 codes per octave)
GAIN_BITS = 7
GAIN_Q = 8
GAIN_BIAS = 27  # code 59 / gain 16 / 128 kbps-ish with rANS
GAIN_MAX_CODE = (1 << GAIN_BITS) - 1  # 127
GAIN_RC_MAX = GAIN_BIAS + GAIN_Q * 6  # cap: gain = 64 (6 octaves above unity)

NF_SEED_BIAS = 0x9E3779B9
NF_START_BIN = 75  # ~3.5 kHz, below this the holes are left alone
NF_OCCUPANCY_STEPS = 32  # resolution of the zero-fraction to level table

# NF cliff rule, see quant.c in c
NF_CLIFF_FIRST_BAND = 17
NF_CLIFF_MIN_DROP = 4


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


def _laplace_rms(z):
    """Sub-deadzone RMS (in units of step) of a Laplacian with zero fraction z"""
    L = -math.log(1.0 - z)
    return (DEAD_ZONE / L) * math.sqrt(2.0 - (1.0 - z) * (L * L + 2.0 * L + 2.0))


# Level per zero-occupancy bucket, sampled at the bucket centers.
# The C reference stores this rounded to Q8 (quant.c nf_laplace_rms_q8).
NF_LAPLACE_RMS = [
    _laplace_rms((k + 0.5) / NF_OCCUPANCY_STEPS) for k in range(NF_OCCUPANCY_STEPS)
]


def _cliff_fill_step(exp_indices, b, gain):
    """Fill step across a spectral cliff, or None when the band is not past one.
    See quant.c in C for details
    """
    if b < NF_CLIFF_FIRST_BAND:
        return None
    drop = int(exp_indices[b - 1]) - int(exp_indices[b])
    if drop <= NF_CLIFF_MIN_DROP:
        return None
    fill_exp = max(
        EXP_INDEX_MIN,
        int(exp_indices[b]) - 3 * (drop - NF_CLIFF_MIN_DROP) // 4,
    )
    return exp_to_step_factor(fill_exp) / gain


def noise_fill(X_hat, q, step, seed, exp_indices, gain, skip_bands=None):
    """Replace zeroed bins above NF_START_BIN with pseudorandom noise, in place.

    The fill level comes from each band's zero occupancy, scaled by the excess of zeros over nonzeros.

    skip_bands (bool per band) leaves bands as exact zeros, used with M/S flags
    """
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        # NF_START_BIN sits on a band edge, so partial bands never occur
        if s < NF_START_BIN or (skip_bands is not None and skip_bands[b]):
            continue
        cliff_step = _cliff_fill_step(exp_indices, b, gain)

        zeros = [i for i in range(s, e) if q[i] == 0]
        if not zeros:
            continue

        # Only the excess of zeros over nonzeros counts as evidence of a missing
        # noise floor. Every zero bin is filled at that excess
        z = len(zeros) / float(e - s)
        excess = 2.0 * z - 1.0
        if excess <= 0.0:
            # less than 50% is zeros - skip NF, enough spectral information is present
            continue

        # All-zero bands land in the last bucket
        zi = min(int(z * NF_OCCUPANCY_STEPS), NF_OCCUPANCY_STEPS - 1)
        factor = NF_LAPLACE_RMS[zi] * excess

        for i in zeros:
            # xorshift
            seed ^= (seed << 13) & 0xFFFFFFFF
            seed ^= seed >> 17
            seed ^= (seed << 5) & 0xFFFFFFFF

            sign = -1.0 if (seed & 0x80000000) else 1.0
            X_hat[i] = sign * factor * (step[i] if cliff_step is None else cliff_step)

    return seed
