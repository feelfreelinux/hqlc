"""Scalar quantization with deadzone and band-weighted step sizes.

The quantizer maps MDCT coefficients to integer symbols:
  q = sign(x) * floor(|x|/step - dz + 1)   if |x|/step > dz
  q = 0                                      otherwise

Dequantization uses an MMSE-optimal centroid offset:
  x_hat = sign(q) * (|q| + centroid) * step

The step size is: step = 2^((exp_idx - 43) / 4) * BW[b] / gain
  - exp_idx: per-band energy (6-bit log scale)
  - BW[b]:   band weight = log2(width) / log2(max_width), normalizes precision
  - gain:    global gain code (7-bit, 8 codes per octave)
"""

import math

import numpy as np

from .constants import (
    BAND_WEIGHTS,
    CENTROID,
    DEAD_ZONE,
    EXP_INDEX_BIAS,
)

# Gain quantization: 7-bit code, log2 Q3 (8 codes per octave)
GAIN_BITS = 7
GAIN_Q = 8
GAIN_BIAS = 48  # code for gain = 1.0
GAIN_MAX_CODE = (1 << GAIN_BITS) - 1  # 127
GAIN_RC_MAX = GAIN_BIAS + GAIN_Q * 3  # cap: gain = 8 (3 octaves above unity)


def quantize_gain(gain):
    """Encode a gain value to a 7-bit code."""
    code = int(round(math.log2(max(1e-12, gain)) * GAIN_Q)) + GAIN_BIAS
    return max(0, min(GAIN_MAX_CODE, code))


def dequantize_gain(code):
    """Decode a 7-bit gain code to a gain value."""
    return 2.0 ** ((code - GAIN_BIAS) / float(GAIN_Q))


def quantize_band(coeffs, exp_idx, inv_gain, band_idx):
    """Deadzone-quantize one band of MDCT coefficients."""
    qquarter = exp_idx - EXP_INDEX_BIAS
    step = (2.0 ** (qquarter / 4.0)) * BAND_WEIGHTS[band_idx] * inv_gain
    scaled = coeffs / step
    abs_sc = np.abs(scaled)
    q = np.zeros(len(coeffs), dtype=np.int32)
    mask = abs_sc > DEAD_ZONE
    q[mask] = (np.sign(scaled[mask]) * np.floor(abs_sc[mask] - DEAD_ZONE + 1.0)).astype(
        np.int32
    )
    return q


def dequantize_band(q, exp_idx, inv_gain, band_idx):
    """Inverse quantize one band with centroid reconstruction."""
    qquarter = exp_idx - EXP_INDEX_BIAS
    step = (2.0 ** (qquarter / 4.0)) * BAND_WEIGHTS[band_idx] * inv_gain
    y = np.zeros(len(q), dtype=np.float64)
    nz = q != 0
    y[nz] = np.sign(q[nz]) * (np.abs(q[nz]) + CENTROID) * step
    return y
