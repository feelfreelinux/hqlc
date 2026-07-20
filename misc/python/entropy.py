"""
Implements an rANS streaming encoder and some rice coding utilities.

An MDCT coefficient is coded as:
  - magnitude symbol: 0..14 literal, 15 = escape
  - optional Exp-Golomb(0) overflow for magnitudes >= 15
  - optional sign bit (equivalent probability)

Table selection is alpha x activity, in order to capture per-band and per-gain variation
  - 12 alpha bins x 4 activity bins = 48 tables
  - Alpha: log2(gain * sigma_pair), covers gain and band-pair variation
  - Activity: quantized non-zero fraction of the previous band (0-3)
  - Both are available to the decoder for free, no cross-frame state

Side information (exponents, TNS, noise factor) uses Rice or fixed-width fields
"""

import math

import numpy as np

from .constants import BAND_EDGES

# rANS constants
RANS_M = 1024
RANS_MAX_SYM = 16  # 0..14 magnitudes + 15 ESC
RANS_L = 1 << 16

# Sign bit has equivalent probability
_SIGN_FREQ = [512, 512]
_SIGN_CF = [0, 512, 1024]


# Bit stream I/O
class BitWriter:
    """Pack bits into bytes, MSB first."""

    __slots__ = ("_buf", "_byte", "_n")

    def __init__(self):
        self._buf = bytearray()
        self._byte = 0
        self._n = 0

    def write(self, value, nbits):
        for i in range(nbits - 1, -1, -1):
            self._byte = (self._byte << 1) | ((value >> i) & 1)
            self._n += 1
            if self._n == 8:
                self._buf.append(self._byte)
                self._byte = 0
                self._n = 0

    def write_rice(self, val, k):
        q = val >> k
        for _ in range(q):
            self.write(1, 1)
        self.write(0, 1)
        if k > 0:
            self.write(val & ((1 << k) - 1), k)

    def total_bits(self):
        return len(self._buf) * 8 + self._n

    def flush(self):
        if self._n > 0:
            self._byte <<= 8 - self._n
            self._buf.append(self._byte)
            self._byte = 0
            self._n = 0

    def get_bytes(self):
        self.flush()
        return bytes(self._buf)


class BitReader:
    """Read bits from bytes, MSB first."""

    __slots__ = ("_data", "_pos", "_bit")

    def __init__(self, data):
        self._data = data
        self._pos = 0
        self._bit = 0

    def read(self, nbits):
        value = 0
        for _ in range(nbits):
            if self._pos >= len(self._data):
                return value << (nbits - _)  # zero-pad on overread
            b = (self._data[self._pos] >> (7 - self._bit)) & 1
            value = (value << 1) | b
            self._bit += 1
            if self._bit == 8:
                self._pos += 1
                self._bit = 0
        return value

    def read_rice(self, k):
        q = 0
        while self.read(1) and self._pos < len(self._data):
            q += 1
        return (q << k) | (self.read(k) if k > 0 else 0)

    def bits_read(self):
        return self._pos * 8 + self._bit


class RANSEncoder:
    """rANS encoder: M=1024, single stream"""

    __slots__ = ("_syms",)

    def __init__(self):
        self._syms = []

    def put(self, s, freq, cumfreq):
        self._syms.append((s, freq, cumfreq))

    def finish(self):
        if not self._syms:
            return b""
        out = []
        state = RANS_L
        # Encode in reverse so the decoder reads symbols in order (ANS is LIFO).
        for s, freq, cumfreq in reversed(self._syms):
            f = freq[s]
            # Renormalization flushes low bytes so the encoding step won't overflow.
            while state >= f << 14:
                out.append(state & 0xFF)
                state >>= 8
            # Fold symbol s into the state: state = (state/f)*M + state%f + cf.
            q, r = divmod(state, f)
            state = q * RANS_M + r + cumfreq[s]
        for _ in range(4):  # flush the final 4-byte state
            out.append(state & 0xFF)
            state >>= 8
        out.reverse()
        return bytes(out)


class RANSDecoder:
    """rANS decoder: M=1024, single stream"""

    __slots__ = ("state", "_data", "_pos")

    def __init__(self, data):
        self._data = data
        self._pos = 0
        self.state = 0
        # Get state from the encoder's 4-byte flush
        for _ in range(4):
            self.state = (self.state << 8) | self._data[self._pos]
            self._pos += 1

    def get(self, freq, cumfreq):
        slot = self.state % RANS_M  # low bits pick the symbol via its cf range
        s = 0
        while s < len(freq) - 1 and cumfreq[s + 1] <= slot:
            s += 1
        f = freq[s]
        # Inverse of the encoding fold, then renormalize by pulling bytes back
        self.state = f * (self.state // RANS_M) + slot - cumfreq[s]
        while self.state < RANS_L and self._pos < len(self._data):
            self.state = (self.state << 8) | self._data[self._pos]
            self._pos += 1
        return s


# Alpha x activity rANS tables (12 x 4 = 48), trained on MUSDB18 and SQAM
_RANS_ALPHA_NBINS = 12
_RANS_ACT_NBINS = 4
_RANS_NTABLES = 48
_RANS_ALPHA_LO = -4.965784284662087
_RANS_ALPHA_HI = 5.372394965238024
_RANS_PAIR_SIGMA = [
    0.863,
    0.527,
    0.238,
    0.117,
    0.073,
    0.051,
    0.030,
    0.021,
    0.011,
    0.005,
]
_RANS_FREQ = [
    [1009, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=0
    [1007, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=1
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  # a=0 act=2
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  # a=0 act=3
    [995, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=0
    [940, 70, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=1
    [917, 93, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=2
    [759, 245, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=3
    [968, 42, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=0
    [906, 104, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=1
    [833, 171, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=2
    [770, 226, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=3
    [972, 38, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=0
    [828, 172, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=1
    [729, 255, 27, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=2
    [671, 290, 47, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=3
    [993, 16, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=0
    [737, 220, 40, 12, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=1
    [604, 332, 64, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=2
    [504, 393, 99, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=3
    [1009, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=0
    [646, 207, 75, 37, 22, 13, 8, 5, 3, 2, 1, 1, 1, 1, 1, 1],  # a=5 act=1
    [494, 320, 122, 46, 19, 9, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=2
    [380, 380, 171, 60, 18, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=3
    [1009, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=6 act=0
    [572, 199, 84, 49, 31, 22, 16, 12, 9, 7, 5, 4, 3, 2, 2, 7],  # a=6 act=1
    [436, 270, 130, 72, 41, 25, 16, 10, 7, 5, 3, 2, 2, 1, 1, 3],  # a=6 act=2
    [271, 305, 193, 114, 63, 34, 18, 10, 6, 3, 2, 1, 1, 1, 1, 1],  # a=6 act=3
    [1009, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=7 act=0
    [453, 189, 89, 58, 40, 26, 21, 16, 15, 12, 12, 11, 9, 7, 8, 58],  # a=7 act=1
    [339, 235, 126, 82, 57, 41, 29, 22, 17, 13, 10, 8, 7, 6, 5, 27],  # a=7 act=2
    [185, 222, 166, 125, 92, 67, 49, 34, 24, 17, 12, 8, 6, 4, 3, 10],  # a=7 act=3
    [765, 37, 28, 23, 21, 18, 17, 15, 13, 11, 10, 9, 8, 7, 6, 36],  # a=8 act=0
    [310, 182, 101, 73, 58, 45, 29, 22, 21, 11, 12, 11, 8, 9, 8, 124],  # a=8 act=1
    [223, 201, 128, 90, 69, 53, 43, 33, 26, 21, 16, 14, 11, 9, 8, 79],  # a=8 act=2
    [135, 171, 138, 112, 91, 73, 59, 47, 38, 30, 24, 19, 15, 12, 10, 50],  # a=8 act=3
    [597, 41, 36, 31, 26, 24, 22, 20, 19, 18, 17, 15, 14, 13, 12, 119],  # a=9 act=0
    [105, 157, 92, 130, 66, 53, 26, 13, 1, 13, 26, 1, 26, 26, 1, 288],  # a=9 act=1
    [109, 119, 89, 74, 66, 53, 47, 40, 35, 33, 30, 25, 22, 23, 19, 240],  # a=9 act=2
    [78, 109, 97, 86, 77, 68, 61, 54, 48, 42, 37, 32, 28, 25, 21, 161],  # a=9 act=3
    [382, 42, 40, 36, 32, 29, 26, 24, 23, 22, 21, 19, 19, 18, 17, 274],  # a=10 act=0
    [1, 136, 136, 33, 67, 136, 67, 170, 1, 1, 34, 1, 1, 1, 1, 238],  # a=10 act=1
    [45, 68, 55, 45, 44, 45, 37, 39, 29, 23, 26, 22, 21, 21, 19, 485],  # a=10 act=2
    [45, 66, 62, 58, 54, 51, 48, 45, 42, 40, 37, 35, 32, 30, 28, 351],  # a=10 act=3
    [207, 28, 29, 28, 27, 25, 23, 22, 21, 19, 18, 18, 17, 16, 16, 510],  # a=11 act=0
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  # a=11 act=1
    [20, 43, 38, 36, 25, 20, 14, 34, 20, 25, 23, 27, 25, 9, 9, 656],  # a=11 act=2
    [24, 36, 35, 34, 33, 31, 30, 29, 28, 27, 26, 26, 25, 24, 23, 593],  # a=11 act=3
]
_RANS_CF = [None] * _RANS_NTABLES
_RANS_COST_Q8 = [None] * _RANS_NTABLES


def _cumfreq(freq):
    """Build cumulative frequency list from frequency array."""
    cf = [0] * (len(freq) + 1)
    for i in range(len(freq)):
        cf[i + 1] = cf[i] + int(freq[i])
    return cf


# Pre-build CFs and cost tables
for _bk in range(_RANS_NTABLES):
    _RANS_CF[_bk] = _cumfreq(_RANS_FREQ[_bk])
    _RANS_COST_Q8[_bk] = [
        int(round(-math.log2(max(f, 1) / RANS_M) * 256.0)) for f in _RANS_FREQ[_bk]
    ]


def _alpha_bin(band, gain):
    """Compute alpha bin from band and gain. Decoder-symmetric."""
    pair = band // 2
    alpha = gain * _RANS_PAIR_SIGMA[pair]
    la = math.log2(max(alpha, 1e-6))
    idx = int(
        (la - _RANS_ALPHA_LO) / (_RANS_ALPHA_HI - _RANS_ALPHA_LO) * _RANS_ALPHA_NBINS
    )
    return max(0, min(_RANS_ALPHA_NBINS - 1, idx))


def _prev_band_activity(q_flat, band):
    """Compute activity bin from previous band's decoded coefficients

    Returns 0-3 based on non-zero fraction of band-1
    """
    if band == 0:
        return 0
    s = BAND_EDGES[band - 1]
    e = BAND_EDGES[band]
    w = e - s
    nz = int(np.count_nonzero(q_flat[s:e]))
    frac = nz / w
    if frac < 0.1:
        return 0
    elif frac < 0.3:
        return 1
    elif frac < 0.6:
        return 2
    else:
        return 3


def rans_table_idx(band, gain, activity):
    """Compute the rANS probability table index.

    table_idx = alpha_bin * N_ACT + activity_bin.
    Both alpha and activity are decoder-symmetric.
    """
    abin = _alpha_bin(band, gain)
    return abin * _RANS_ACT_NBINS + activity


def get_rans_tables(table_idx):
    """Return (freq, cf, cost_q8) for a rANS probability table index."""
    return (
        _RANS_FREQ[table_idx],
        _RANS_CF[table_idx],
        _RANS_COST_Q8[table_idx],
    )


def coeff_cost_q8(cost_q8, value):
    """Cost in Q8 bits for one signed quantized coefficient."""
    mag = abs(value)
    if mag < RANS_MAX_SYM - 1:
        c = int(cost_q8[mag])
    else:
        c = int(cost_q8[RANS_MAX_SYM - 1])
        overflow = mag - (RANS_MAX_SYM - 1)
        nbits = (overflow + 1).bit_length() - 1
        c += (2 * nbits + 1) * 256  # EG(0) prefix + suffix
    if value != 0:
        c += 256  # sign bit
    return c


# Rice helpers
def zigzag_enc(val):
    return ((-val << 1) - 1) if val < 0 else (val << 1)


def zigzag_dec(u):
    return -((u + 1) >> 1) if (u & 1) else (u >> 1)


def find_best_rice_k(deltas):
    """Find the k that minimizes total Rice-coded bits."""
    best_cost, best_k = float("inf"), 0
    for k in range(7):
        cost = sum((zigzag_enc(d) >> k) + 1 + k for d in deltas)
        if cost < best_cost:
            best_cost, best_k = cost, k
    return best_k
