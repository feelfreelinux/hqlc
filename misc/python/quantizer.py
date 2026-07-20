"""Scalar quantization: gain code, per-band step size, deadzone quantizer.

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

from .constants import CENTROID, DEAD_ZONE, EXP_INDEX_BIAS

# Gain quantization: 7-bit code, log2 Q3 (8 codes per octave)
GAIN_BITS = 7
GAIN_Q = 8
GAIN_BIAS = 27  # code 59 / gain 16 / 128 kbps-ish with rANS
GAIN_MAX_CODE = (1 << GAIN_BITS) - 1  # 127
GAIN_RC_MAX = GAIN_BIAS + GAIN_Q * 5  # cap: gain = 32 (5 octaves above unity)


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

    Returns (q, abs_scaled): the integer symbols and the pre-deadzone
    magnitudes |X / step|, which the noise-fill estimator reuses. step
    fixes the length; X is sliced to match (the inaudible HF tail is dropped).
    """
    n = len(step)
    scaled = np.zeros(n)
    valid = step > 1e-20  # guard against divide-by-zero on empty bands
    scaled[valid] = X[:n][valid] / step[valid]
    abs_scaled = np.abs(scaled)

    q = np.zeros(n, dtype=np.int32)
    mask = abs_scaled > DEAD_ZONE
    q[mask] = (np.sign(scaled[mask])
               * np.floor(abs_scaled[mask] - DEAD_ZONE + 1.0)).astype(np.int32)
    return q, abs_scaled


def dequantize(q, step):
    """Centroid reconstruction: x_hat = sign(q) * (|q| + CENTROID) * step."""
    x = np.zeros(len(step))
    nz = q != 0
    x[nz] = np.sign(q[nz]) * (np.abs(q[nz]) + CENTROID) * step[nz]
    return x
