"""Core constants and band definitions for HQLC."""

import math

# Frame / block dimensions
FRAME_LEN = 512
BLOCK_SIZE = 1024
N_BINS = 512
FS = 48000

# 20 non-uniform bands, roughly ERB-spaced (wider in LF, narrower after 2 kHz)
# Stops at bin 427 (~20 kHz), higher bands are zeroed
N_BANDS = 20
BAND_EDGES = [
    0,
    3,
    8,
    13,
    19,
    26,
    34,
    43,
    52,
    62,
    75,
    89,
    107,
    127,
    152,
    180,
    215,
    255,
    303,
    360,
    427,
]

# 48 fine bands for exponent computation
#
# Single-bin bands below 844 Hz (bins 0-17) to give more resolution there,
# while higher bands follow ERB spacing
# Generated with plateau_erb_edges(n_single=18, fs=48000, n_bins=512)
FINE_BAND_EDGES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    20,
    23,
    26,
    29,
    33,
    37,
    42,
    47,
    53,
    59,
    66,
    74,
    83,
    92,
    102,
    114,
    127,
    141,
    157,
    174,
    193,
    214,
    238,
    264,
    293,
    325,
    361,
    400,
    427,
    N_BINS,
]

N_FINE_BANDS = len(FINE_BAND_EDGES) - 1
N_ACTIVE_FINE = N_FINE_BANDS - 1

# Pre-emphasis tilt (dB) that is baked into the exponent computation, does static perceptual shaping
FINE_TILT_DB = 35.0


# Precomputed fine band to coarse mapping
_FINE_CENTERS = [
    (FINE_BAND_EDGES[i] + FINE_BAND_EDGES[i + 1]) / 2.0 for i in range(N_FINE_BANDS)
]
FB_COARSE = []
for _fc in _FINE_CENTERS:
    for _b in range(N_BANDS):
        if BAND_EDGES[_b] <= _fc < BAND_EDGES[_b + 1]:
            FB_COARSE.append(_b)
            break
    else:
        FB_COARSE.append(N_BANDS - 1)

# Exponent value is a log-domain energy descriptor, used to scale the quantizer accordingly
# step = 2^((idx - BIAS) / 4)
EXP_INDEX_BIAS = 43
EXP_INDEX_MAX = 63
EXP_INDEX_MIN = 0

# One exponent index is a quarter octave of step size, so ~1.505 dB
DB_PER_EXP_INDEX = 20.0 * math.log10(2.0) / 4.0

# Quantizer parameters
DEAD_ZONE = 0.65  # below this threshold, coefficient quantizes to zero

# Reconstruction offset for the quantizer
CENTROID = 0.15
