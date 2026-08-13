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


def write_binary_rle(w, flags, rice_k=1):
    """Run-length code a bool sequence, bit-identical to C bw_write_binary_rle
    """
    n = len(flags)
    if n <= 0:
        return
    w.write(1 if flags[0] else 0, 1)  # initial color
    pos = 0
    while pos < n:
        run = 1
        while pos + run < n and bool(flags[pos + run]) == bool(flags[pos]):
            run += 1
        if pos + run < n:  # another run follows
            w.write(1, 1)
            w.write_rice(run - 1, rice_k)
        pos += run
    w.write(0, 1)  # terminator


def read_binary_rle(r, n_flags, rice_k=1):
    """Inverse of write_binary_rle, matching C br_read_binary_rle."""
    flags = [False] * n_flags
    if n_flags <= 0:
        return flags
    cur = bool(r.read(1))  # initial color
    pos = 0
    for _ in range(n_flags):
        if pos >= n_flags or not r.read(1):
            break
        run = r.read_rice(rice_k) + 1
        for _ in range(run):
            if pos >= n_flags:
                break
            flags[pos] = cur
            pos += 1
        cur = not cur
    while pos < n_flags:  # trailing run implied by the terminator
        flags[pos] = cur
        pos += 1
    return flags


def binary_rle_bits(flags, rice_k=1):
    """Exact encoded bit length of write_binary_rle(flags, rice_k)."""
    w = BitWriter()
    write_binary_rle(w, flags, rice_k)
    return w.total_bits()


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
    [976, 33, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=0
    [959, 49, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=1
    [963, 45, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=2
    [942, 63, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=0 act=3
    [951, 57, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=0
    [909, 96, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=1
    [911, 94, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=2
    [854, 151, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=1 act=3
    [914, 89, 7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=0
    [857, 142, 10, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=1
    [809, 173, 24, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=2
    [781, 201, 24, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=2 act=3
    [877, 118, 13, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=0
    [775, 216, 18, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=1
    [708, 270, 29, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=2
    [641, 301, 57, 11, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=3 act=3
    [874, 108, 22, 6, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=0
    [667, 277, 52, 12, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=1
    [590, 341, 66, 12, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=2
    [513, 378, 97, 19, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=4 act=3
    [937, 46, 17, 7, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=0
    [565, 288, 97, 37, 15, 7, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=1
    [486, 346, 120, 39, 14, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=2
    [408, 375, 153, 52, 17, 6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],  # a=5 act=3
    [902, 39, 24, 17, 12, 8, 6, 4, 3, 2, 2, 1, 1, 1, 1, 1],  # a=6 act=0
    [485, 274, 120, 60, 33, 19, 11, 7, 4, 3, 2, 2, 1, 1, 1, 1],  # a=6 act=1
    [403, 314, 151, 73, 36, 18, 10, 6, 3, 2, 2, 2, 1, 1, 1, 1],  # a=6 act=2
    [316, 332, 188, 95, 45, 21, 10, 5, 3, 2, 2, 1, 1, 1, 1, 1],  # a=6 act=3
    [526, 149, 97, 71, 52, 37, 27, 18, 13, 9, 6, 5, 3, 3, 2, 6],  # a=7 act=0
    [404, 247, 124, 76, 49, 34, 24, 17, 13, 9, 7, 5, 4, 3, 3, 5],  # a=7 act=1
    [321, 269, 153, 95, 61, 39, 26, 17, 12, 8, 6, 5, 3, 3, 2, 4],  # a=7 act=2
    [224, 263, 184, 124, 81, 52, 33, 21, 13, 9, 6, 4, 3, 2, 2, 3],  # a=7 act=3
    [419, 96, 75, 62, 53, 46, 40, 35, 31, 27, 23, 20, 17, 14, 12, 54],  # a=8 act=0
    [292, 236, 139, 93, 65, 47, 33, 26, 19, 16, 13, 9, 7, 6, 5, 18],  # a=8 act=1
    [230, 230, 150, 104, 76, 56, 41, 31, 23, 18, 13, 10, 8, 7, 5, 22],  # a=8 act=2
    [153, 198, 157, 122, 94, 72, 54, 40, 30, 23, 18, 13, 10, 8, 7, 25],  # a=8 act=3
    [353, 75, 64, 54, 47, 42, 38, 35, 32, 29, 27, 24, 22, 20, 18, 144],  # a=9 act=0
    [222, 202, 129, 86, 73, 47, 40, 34, 26, 25, 19, 16, 15, 14, 15, 61],  # a=9 act=1
    [151, 175, 127, 97, 77, 63, 52, 43, 36, 29, 26, 21, 18, 16, 13, 80],  # a=9 act=2
    [95, 133, 117, 102, 88, 75, 65, 55, 46, 39, 32, 27, 23, 19, 16, 92],  # a=9 act=3
    [251, 46, 44, 41, 38, 35, 33, 31, 29, 27, 26, 24, 23, 22, 21, 333],  # a=10 act=0
    [292, 118, 51, 101, 51, 26, 43, 26, 18, 26, 18, 34, 26, 18, 9, 167],  # a=10 act=1
    [42, 56, 54, 51, 48, 46, 41, 41, 39, 37, 37, 35, 33, 31, 29, 404],  # a=10 act=2
    [49, 72, 70, 66, 63, 59, 55, 52, 48, 44, 41, 38, 34, 31, 29, 273],  # a=10 act=3
    [56, 34, 33, 33, 31, 30, 29, 27, 26, 25, 24, 23, 22, 21, 21, 589],  # a=11 act=0
    [1009, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # a=11 act=1
    [18, 28, 31, 28, 28, 29, 27, 30, 25, 26, 23, 20, 26, 23, 23, 639],  # a=11 act=2
    [25, 38, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 29, 27, 26, 544],  # a=11 act=3
]
_RANS_CF = [None] * _RANS_NTABLES
_RANS_COST_BITS = [None] * _RANS_NTABLES


def _cumfreq(freq):
    """Build cumulative frequency list from frequency array."""
    cf = [0] * (len(freq) + 1)
    for i in range(len(freq)):
        cf[i + 1] = cf[i] + int(freq[i])
    return cf


# Pre-build CFs and cost tables
for _bk in range(_RANS_NTABLES):
    _RANS_CF[_bk] = _cumfreq(_RANS_FREQ[_bk])
    _RANS_COST_BITS[_bk] = [-math.log2(max(f, 1) / RANS_M) for f in _RANS_FREQ[_bk]]


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


def rans_table_idx(band, gain, activity, alpha_shift=0):
    """Compute the rANS probability table index.

    table_idx = alpha_bin * N_ACT + activity_bin

    alpha_shift coarsens the alpha for S channel in M/S flagged bands
    """
    abin = _alpha_bin(band, gain)
    if alpha_shift:
        abin = max(0, abin - alpha_shift)
    return abin * _RANS_ACT_NBINS + activity


def get_rans_tables(table_idx):
    """Return (freq, cf, cost_bits) for a rANS probability table index."""
    return (
        _RANS_FREQ[table_idx],
        _RANS_CF[table_idx],
        _RANS_COST_BITS[table_idx],
    )


def coeff_cost(cost_bits, value):
    """Cost in bits for one signed quantized coefficient."""
    mag = abs(value)
    if mag < RANS_MAX_SYM - 1:
        c = cost_bits[mag]
    else:
        c = cost_bits[RANS_MAX_SYM - 1]
        overflow = mag - (RANS_MAX_SYM - 1)
        nbits = (overflow + 1).bit_length() - 1
        c += 2 * nbits + 1  # EG(0) prefix + suffix
    if value != 0:
        c += 1  # sign bit
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
