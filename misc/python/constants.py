"""Core constants, band definitions, and band weights for HQLC."""

import math

import numpy as np

# Frame / block dimensions
FRAME_LEN = 512
BLOCK_SIZE = 1024
N_BINS = 512
FS = 48000

# 20 non-uniform bands, approximately ERB-spaced
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
    55,
    67,
    82,
    101,
    123,
    149,
    179,
    214,
    255,
    304,
    362,
    431,
    512,
]

# Exponent index: log-domain energy descriptor per band
# step = 2^((idx - BIAS) / 4), giving ~1.5 dB granularity
EXP_INDEX_BIAS = 43
EXP_INDEX_MAX = 63
EXP_INDEX_MIN = 0

# Quantizer parameters
DEAD_ZONE = 0.65  # below this threshold, coefficient quantizes to zero
CENTROID = 0.15  # MMSE-optimal reconstruction offset for Laplacian source

# Band weights: BW[b] = log2(bin_count[b]) / log2(max_bin_count), clipped [0.10, 1.0]
# Normalizes quantizer precision across bands of different widths
_BIN_COUNTS = np.array(
    [BAND_EDGES[b + 1] - BAND_EDGES[b] for b in range(N_BANDS)], dtype=np.float64
)
_LOG2_BINS = np.log2(_BIN_COUNTS)
BAND_WEIGHTS = np.clip(_LOG2_BINS / np.max(_LOG2_BINS), 0.10, 1.0)

# log2(bin_count) in Q4 per band (for PSD to total energy conversion)
LOG2_BINS_Q4 = np.array(
    [
        int(round(16.0 * math.log2(max(1, BAND_EDGES[b + 1] - BAND_EDGES[b]))))
        for b in range(N_BANDS)
    ],
    dtype=np.int32,
)
