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
        for _ in range(3):  # flush the final 3-byte state (renorm keeps it < 2**24)
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
        # Get state from the encoder's 3-byte flush
        for _ in range(3):
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


# Alpha x activity rANS tables (12 x 4 = 48), trained 2026-07 on MUSDB18-train,
# SQAM, and personal material at 64/96/128/196 kbps RC (pooled equally per rate)
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
    [998, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=0
    [993, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=1
    [827, 174, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=2
    [124, 77, 64, 60, 59, 59, 59, 58, 58, 58, 58, 58, 58, 58, 58, 58],  # a=0 act=3
    [959, 51, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=0
    [954, 56, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=1
    [885, 123, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=2
    [640, 311, 44, 13, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=3
    [913, 96, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=0
    [865, 143, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=1
    [834, 168, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=2
    [740, 254, 17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=3
    [906, 95, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=0
    [781, 217, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=1
    [705, 284, 22, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=2
    [645, 331, 35, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=3
    [930, 65, 14, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=0
    [723, 238, 42, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=1
    [620, 329, 56, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=2
    [549, 379, 75, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=3
    [969, 31, 8, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=0
    [659, 245, 70, 25, 10, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=1
    [534, 340, 102, 28, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=2
    [457, 386, 130, 33, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=3
    [974, 21, 8, 5, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=6 act=0
    [567, 234, 93, 48, 29, 18, 12, 7, 5, 3, 2, 1, 1, 1, 1, 2],  # a=6 act=1
    [440, 311, 139, 65, 32, 16, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1],  # a=6 act=2
    [344, 344, 180, 84, 38, 16, 7, 3, 1, 1, 1, 1, 1, 1, 1, 1],  # a=6 act=3
    [487, 147, 109, 84, 63, 45, 30, 19, 12, 8, 5, 3, 2, 2, 1, 7],  # a=7 act=0
    [545, 198, 79, 45, 30, 23, 18, 15, 13, 11, 9, 8, 6, 5, 4, 15],  # a=7 act=1
    [370, 253, 136, 85, 54, 36, 25, 17, 13, 9, 7, 5, 3, 2, 2, 7],  # a=7 act=2
    [235, 273, 187, 123, 78, 48, 30, 18, 11, 7, 4, 3, 2, 1, 1, 3],  # a=7 act=3
    [309, 92, 80, 69, 61, 55, 49, 44, 39, 34, 30, 26, 22, 19, 16, 79],  # a=8 act=0
    [433, 220, 98, 59, 41, 32, 25, 20, 16, 13, 10, 8, 7, 6, 5, 31],  # a=8 act=1
    [317, 245, 136, 87, 60, 43, 31, 23, 17, 12, 9, 7, 6, 4, 3, 24],  # a=8 act=2
    [185, 218, 161, 119, 88, 65, 47, 35, 26, 19, 14, 10, 8, 6, 4, 19],  # a=8 act=3
    [753, 18, 17, 16, 16, 15, 14, 14, 13, 13, 12, 10, 9, 8, 8, 88],  # a=9 act=0
    [465, 148, 80, 49, 32, 26, 20, 16, 13, 13, 11, 10, 10, 9, 7, 115],  # a=9 act=1
    [301, 207, 118, 75, 55, 42, 33, 27, 22, 19, 16, 14, 12, 10, 9, 64],  # a=9 act=2
    [129, 165, 135, 110, 90, 73, 59, 48, 38, 31, 24, 20, 16, 13, 11, 62],  # a=9 act=3
    [192, 47, 46, 43, 40, 37, 35, 33, 31, 30, 28, 27, 25, 24, 23, 363],  # a=10 act=0
    [139, 121, 118, 29, 58, 116, 57, 144, 1, 1, 29, 1, 1, 1, 1, 207],  # a=10 act=1
    [83, 78, 69, 61, 55, 50, 46, 40, 38, 34, 34, 29, 26, 26, 23, 332],  # a=10 act=2
    [60, 87, 82, 76, 70, 64, 59, 54, 48, 44, 39, 35, 32, 28, 25, 221],  # a=10 act=3
    [106, 33, 32, 32, 31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 21, 542],  # a=11 act=0
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  # a=11 act=1
    [27, 43, 38, 36, 32, 29, 27, 31, 27, 28, 26, 27, 25, 17, 17, 594],  # a=11 act=2
    [29, 44, 42, 41, 40, 39, 38, 37, 36, 35, 33, 32, 31, 30, 29, 488],  # a=11 act=3
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
