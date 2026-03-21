"""Entropy coding: rANS with trained CDFs, Rice coding, bitstream I/O.

rANS (range variant of Asymmetric Numeral Systems) with M=1024 provides
near-optimal compression with a simple streaming implementation. Each
MDCT coefficient is coded as:
  - magnitude symbol: 0..14 literal, 15 = escape
  - optional Exp-Golomb(0) overflow for magnitudes >= 15
  - optional sign bit (equi-probable)

32 pre-trained frequency tables are indexed by alpha = K * BW[b] / gain,
giving the encoder's CDF estimate. Adjacent bands are paired (10 pairs
from 20 bands) to stabilize statistics.

Side information (exponents, TNS, NF masks) uses Rice coding with an
optimal k parameter selected per block.
"""

import bisect
import math

import numpy as np

from .constants import BAND_EDGES, BAND_WEIGHTS, N_BANDS

# rANS constants
RANS_M = 1024
RANS_MAX_SYM = 16  # 0..14 magnitudes + 15 ESC
RANS_L = 1 << 16
_RANS_K = 1.65

_SIGN_FREQ = [512, 512]
_SIGN_CF = [0, 512, 1024]

# 32-bin alpha-indexed LUT
_RANS_LUT_NBINS = 32

_RANS_LUT_ALPHA_EDGES = [
    0.120000, 0.137334, 0.157171, 0.179874, 0.205856, 0.235591, 0.269622,
    0.308568, 0.353140, 0.404150, 0.462528, 0.529339, 0.605800, 0.693306,
    0.793452, 0.908063, 1.039230, 1.189344, 1.361141, 1.557754, 1.782767,
    2.040282, 2.334995, 2.672277, 3.058280, 3.500039, 4.005609, 4.584207,
    5.246381, 6.004205, 6.871494, 7.864060, 9.000000,
]

_RANS_LUT_FREQ = [
    [82, 113, 100, 88, 77, 68, 60, 53, 46, 41, 36, 31, 28, 24, 21, 156],
    [93, 127, 110, 95, 82, 71, 61, 53, 45, 39, 34, 29, 25, 22, 19, 119],
    [95, 119, 93, 77, 66, 59, 52, 47, 43, 40, 37, 35, 34, 33, 32, 162],
    [120, 159, 130, 108, 89, 73, 60, 50, 41, 34, 28, 23, 19, 16, 13, 61],
    [127, 150, 110, 89, 75, 64, 56, 50, 45, 41, 39, 36, 34, 31, 26, 51],
    [151, 171, 126, 101, 83, 70, 60, 52, 46, 41, 35, 30, 22, 12, 7, 17],
    [173, 199, 146, 115, 92, 73, 58, 46, 36, 28, 21, 16, 10, 6, 3, 2],
    [182, 202, 147, 114, 90, 71, 57, 46, 38, 30, 18, 8, 5, 4, 3, 9],
    [207, 227, 164, 122, 91, 68, 50, 36, 25, 14, 8, 5, 3, 1, 1, 2],
    [225, 253, 183, 129, 89, 59, 38, 22, 11, 6, 3, 2, 1, 1, 1, 1],
    [242, 273, 195, 130, 81, 47, 26, 13, 7, 3, 2, 1, 1, 1, 1, 1],
    [274, 285, 194, 122, 71, 39, 19, 8, 4, 2, 1, 1, 1, 1, 1, 1],
    [310, 311, 191, 107, 56, 26, 10, 4, 2, 1, 1, 1, 1, 1, 1, 1],
    [334, 332, 189, 97, 43, 14, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [374, 351, 178, 75, 26, 8, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [421, 359, 158, 57, 15, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [460, 369, 137, 38, 8, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [517, 366, 108, 19, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [558, 362, 80, 11, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [615, 337, 54, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [674, 302, 33, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [736, 259, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [779, 223, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [834, 173, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [882, 127, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [925, 85, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [953, 57, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [981, 29, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [990, 20, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1002, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1006, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1008, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


# ── Bitstream I/O ──

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


# ── rANS encoder / decoder ──

class RANSEncoder:
    """rANS encoder: M=1024, single stream, per-symbol CDF switching."""
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
        for s, freq, cumfreq in reversed(self._syms):
            f = freq[s]
            while state >= f << 14:
                out.append(state & 0xFF)
                state >>= 8
            q, r = divmod(state, f)
            state = q * RANS_M + r + cumfreq[s]
        for _ in range(4):
            out.append(state & 0xFF)
            state >>= 8
        out.reverse()
        return bytes(out)


class RANSDecoder:
    """rANS decoder: M=1024, single stream, per-symbol CDF switching."""
    __slots__ = ("state", "_data", "_pos")

    def __init__(self, data):
        self._data = data
        self._pos = 0
        self.state = 0
        for _ in range(4):
            self.state = (self.state << 8) | self._data[self._pos]
            self._pos += 1

    def get(self, freq, cumfreq):
        slot = self.state % RANS_M
        s = 0
        while s < len(freq) - 1 and cumfreq[s + 1] <= slot:
            s += 1
        f = freq[s]
        self.state = f * (self.state // RANS_M) + slot - cumfreq[s]
        while self.state < RANS_L and self._pos < len(self._data):
            self.state = (self.state << 8) | self._data[self._pos]
            self._pos += 1
        return s


# ── Table helpers ──

def _cumfreq(freq):
    """Build cumulative frequency list from frequency array."""
    cf = [0] * (len(freq) + 1)
    for i in range(len(freq)):
        cf[i + 1] = cf[i] + int(freq[i])
    return cf


def _lut_lookup(alpha):
    """Look up LUT bin index for given alpha."""
    idx = bisect.bisect_right(_RANS_LUT_ALPHA_EDGES, alpha) - 1
    return max(0, min(_RANS_LUT_NBINS - 1, idx))


def build_band_tables(gain):
    """Build per-band rANS tables for a given gain.

    Adjacent bands are paired. Returns (freq_lists, cf_lists, cost_tables).
    """
    freq_lists = [None] * N_BANDS
    cf_lists = [None] * N_BANDS
    cost_tables = [None] * N_BANDS
    clamped_gain = max(0.1, min(5.0, gain))

    for pi in range(N_BANDS // 2):
        b0, b1 = 2 * pi, 2 * pi + 1
        bw0 = BAND_EDGES[b0 + 1] - BAND_EDGES[b0]
        bw1 = BAND_EDGES[b1 + 1] - BAND_EDGES[b1]
        a0 = _RANS_K * BAND_WEIGHTS[b0] / clamped_gain
        a1 = _RANS_K * BAND_WEIGHTS[b1] / clamped_gain
        pair_alpha = (a0 * bw0 + a1 * bw1) / (bw0 + bw1)
        bin_idx = _lut_lookup(pair_alpha)

        freq_l = _RANS_LUT_FREQ[bin_idx]
        freq = np.array(freq_l, dtype=np.int32)
        cf = _cumfreq(freq)
        cost_q8 = np.array([
            int(round(-math.log2(max(f, 1) / RANS_M) * 256.0)) for f in freq
        ], dtype=np.int32)

        for b in (b0, b1):
            freq_lists[b] = freq_l
            cf_lists[b] = cf
            cost_tables[b] = cost_q8

    return freq_lists, cf_lists, cost_tables


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


# ── Rice coding helpers ──

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
