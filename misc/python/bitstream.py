"""Frame serialization: side info encoding/decoding and bit counting.

Frame layout:
  [gain_code: 7b] [M/S flags: RLE, stereo only] [TNS per ch]
  [padding] [rANS payload]

TNS per channel:
  [active: 1b] [order-1: 3b if active] [LAR indices: 4b each if active]

rANS payload, one stream:
  Exponents first (the decoder needs them before anything else), then the
  coefficients. Per channel the envelope is:
    Ch0: DPCM from the previous band
    Ch1+: [model flag: 1b] then either DPCM or the delta from ch0

+ padding 0-7 bits to byte-align before rANS payload
"""

import numpy as np

from .constants import BAND_EDGES, EXP_INDEX_MAX, EXP_INDEX_MIN, N_BANDS
from .entropy import (
    _SIGN_CF,
    _SIGN_FREQ,
    _prev_band_activity,
    EXP_MODEL_CROSS_CH,
    EXP_MODEL_CROSS_CH_MS,
    EXP_MODEL_DPCM,
    RANS_EXP_NSLOTS,
    RANS_MAX_SYM,
    BitReader,
    BitWriter,
    RANSDecoder,
    RANSEncoder,
    binary_rle_bits,
    eg0_nbits,
    exp_channel_cost_q8,
    get_rans_exp_tables,
    get_rans_tables,
    rans_exp_table_idx,
    rans_table_idx,
    read_binary_rle,
    write_binary_rle,
)
from .ms import MS_RANS_ALPHA_SHIFT
from .quantizer import GAIN_BITS, dequantize_gain, quantize_gain
from .tns import TNS_K_BITS, TNS_LAR_MAX


def count_side_bits(n_ch, tns_orders, tns_q_ks, ms_flags=None):
    """Count side information bits (gain + TNS + pad).

    The exponents are not here, they ride in the rANS payload (see
    entropy.exp_payload_bits for their cost).
    """
    bits = GAIN_BITS
    if ms_flags is not None and n_ch == 2:
        bits += binary_rle_bits(ms_flags, 1)  # per-band M/S flags, binary RLE
    for ch in range(n_ch):
        bits += 1
        if tns_orders[ch] > 0:
            bits += 3 + TNS_K_BITS * tns_orders[ch]

    bits += (8 - bits % 8) % 8  # byte alignment
    return bits


def _exp_model(band, use_dpcm, ms_flags):
    """Pick the rANS model for one ch1+ exponent symbol."""
    if use_dpcm:
        return EXP_MODEL_DPCM  # same model regardless of M/S
    if ms_flags is not None and ms_flags[band]:
        return EXP_MODEL_CROSS_CH_MS
    return EXP_MODEL_CROSS_CH


def _clamp_exp(value):
    """Clamp a decoded exponent index, matching the C decoder."""
    return max(EXP_INDEX_MIN, min(EXP_INDEX_MAX, value))


def _put_exp_symbol(enc, table_idx, value):
    """Queue one centered exponent symbol: magnitude, escape, then sign"""
    freq, cf, center = get_rans_exp_tables(table_idx)
    d = value - center
    mag = abs(d)

    enc.put(min(mag, RANS_EXP_NSLOTS - 1), freq, cf)
    if mag >= RANS_EXP_NSLOTS - 1:
        overflow = mag - (RANS_EXP_NSLOTS - 1)
        nbits = eg0_nbits(overflow)
        for _ in range(nbits):
            enc.put(0, _SIGN_FREQ, _SIGN_CF)
        enc.put(1, _SIGN_FREQ, _SIGN_CF)
        val = overflow + 1
        for bit_idx in range(nbits - 1, -1, -1):
            enc.put((val >> bit_idx) & 1, _SIGN_FREQ, _SIGN_CF)
    if mag != 0:
        enc.put(0 if d > 0 else 1, _SIGN_FREQ, _SIGN_CF)


def _get_exp_symbol(dec, table_idx):
    """Inverse of _put_exp_symbol."""
    freq, cf, center = get_rans_exp_tables(table_idx)

    sym = dec.get(freq, cf)
    mag = sym
    if sym >= RANS_EXP_NSLOTS - 1:
        nbits = 0
        while dec.get(_SIGN_FREQ, _SIGN_CF) == 0:
            nbits += 1
        val = 1
        for _ in range(nbits):
            val = (val << 1) | dec.get(_SIGN_FREQ, _SIGN_CF)
        mag = (RANS_EXP_NSLOTS - 1) + val - 1

    d = mag
    if mag != 0 and dec.get(_SIGN_FREQ, _SIGN_CF):
        d = -mag
    return center + d


def _put_exponents(enc, exp_indices, n_ch, ms_flags=None):
    """Queue the envelope ahead of the coefficients, ch0 first."""
    prev = 0
    for b in range(N_BANDS):
        v = int(exp_indices[0][b])
        _put_exp_symbol(enc, rans_exp_table_idx(EXP_MODEL_DPCM, b), v - prev)
        prev = v

    for ch in range(1, n_ch):
        # Whichever of DPCM / cross-channel is cheaper, signalled in one bit
        _, use_dpcm = exp_channel_cost_q8(exp_indices, ch, ms_flags)
        enc.put(1 if use_dpcm else 0, _SIGN_FREQ, _SIGN_CF)
        prev = 0
        for b in range(N_BANDS):
            v = int(exp_indices[ch][b])
            model = _exp_model(b, use_dpcm, ms_flags)
            ref = prev if use_dpcm else int(exp_indices[0][b])
            _put_exp_symbol(enc, rans_exp_table_idx(model, b), v - ref)
            prev = v


def _get_exponents(dec, n_channels, ms_flags=None):
    """Inverse of _put_exponents."""
    exp_indices = [np.zeros(N_BANDS, dtype=np.int32) for _ in range(n_channels)]

    prev = 0
    for b in range(N_BANDS):
        d = _get_exp_symbol(dec, rans_exp_table_idx(EXP_MODEL_DPCM, b))
        prev = _clamp_exp(prev + d)
        exp_indices[0][b] = prev

    for ch in range(1, n_channels):
        use_dpcm = bool(dec.get(_SIGN_FREQ, _SIGN_CF))
        prev = 0
        for b in range(N_BANDS):
            model = _exp_model(b, use_dpcm, ms_flags)
            d = _get_exp_symbol(dec, rans_exp_table_idx(model, b))
            ref = prev if use_dpcm else int(exp_indices[0][b])
            prev = _clamp_exp(ref + d)
            exp_indices[ch][b] = prev
    return exp_indices


def encode_frame(gain, tns_orders, tns_q_ks, exp_indices, all_quants, ms_flags=None):
    """Encode one frame, returns (total_bits, payload_bytes)"""
    n_ch = len(tns_orders)
    side = BitWriter()

    gain_code = quantize_gain(gain)
    side.write(gain_code, GAIN_BITS)

    # Per-band M/S flags, binary RLE
    if ms_flags is not None:
        write_binary_rle(side, [bool(ms_flags[b]) for b in range(N_BANDS)], 1)

    # TNS
    half = (1 << (TNS_K_BITS - 1)) - 1
    for ch in range(n_ch):
        if tns_orders[ch] == 0:
            side.write(0, 1)
        else:
            side.write(1, 1)
            side.write(tns_orders[ch] - 1, 3)
            for qk in tns_q_ks[ch]:
                side.write(int(qk) + half, TNS_K_BITS)

    # Byte align
    pad = (8 - side.total_bits() % 8) % 8
    if pad:
        side.write(0, pad)
    side_bytes = side.get_bytes()

    # rANS payload: the envelope, then alpha x activity coded coefficients
    gain_dq = dequantize_gain(gain_code)
    rans_enc = RANSEncoder()
    _put_exponents(rans_enc, exp_indices, n_ch, ms_flags)

    active = BAND_EDGES[-1]
    for ch in range(n_ch):
        # Build flat quant array for activity computation
        q_flat = np.zeros(active, dtype=np.int32)
        for b in range(N_BANDS):
            s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
            q_flat[s:e] = all_quants[ch][b]

        for b in range(N_BANDS):
            q = all_quants[ch][b]
            act = _prev_band_activity(q_flat, b)
            ashift = MS_RANS_ALPHA_SHIFT if (ch == 1 and ms_flags is not None and ms_flags[b]) else 0
            tidx = rans_table_idx(b, gain_dq, act, ashift)
            fd, cf, _ = get_rans_tables(tidx)
            for v in q:
                mag = abs(int(v))
                rans_enc.put(min(mag, RANS_MAX_SYM - 1), fd, cf)
                if mag >= RANS_MAX_SYM - 1:
                    overflow = mag - (RANS_MAX_SYM - 1)
                    nbits = eg0_nbits(overflow)
                    for _ in range(nbits):
                        rans_enc.put(0, _SIGN_FREQ, _SIGN_CF)
                    rans_enc.put(1, _SIGN_FREQ, _SIGN_CF)
                    val = overflow + 1
                    for bit_idx in range(nbits - 1, -1, -1):
                        rans_enc.put((val >> bit_idx) & 1, _SIGN_FREQ, _SIGN_CF)
                if int(v) != 0:
                    rans_enc.put(0 if int(v) > 0 else 1, _SIGN_FREQ, _SIGN_CF)

    payload = side_bytes + rans_enc.finish()
    return len(payload) * 8, payload


def decode_frame(payload, n_channels, has_ms=False):
    """Decode one frame from payload bytes.

    ms_flags is None unless has_ms
    Returns (gain, tns_orders, tns_ks, exp_indices, all_quants, ms_flags)
    """
    br = BitReader(payload)
    gain_code = br.read(GAIN_BITS)
    gain = dequantize_gain(gain_code)

    # Per-band M/S flags, binary RLE
    ms_flags = None
    if has_ms:
        ms_flags = read_binary_rle(br, N_BANDS, 1)

    # TNS
    half = (1 << (TNS_K_BITS - 1)) - 1
    tns_orders = []
    tns_q_ks = []
    for ch in range(n_channels):
        if br.read(1):
            order = br.read(3) + 1
            q_k = np.array(
                [br.read(TNS_K_BITS) - half for _ in range(order)], dtype=np.int32
            )
            tns_orders.append(order)
            tns_q_ks.append(q_k)
        else:
            tns_orders.append(0)
            tns_q_ks.append(np.zeros(0, dtype=np.int32))

    # Byte align to rANS start
    bits_used = br.bits_read()
    pad = (8 - bits_used % 8) % 8
    if pad:
        br.read(pad)
    rans_start = br.bits_read() // 8

    rans_data = payload[rans_start:]
    if len(rans_data) == 0:
        raise ValueError("truncated frame: the rANS payload carries the envelope")
    rans_dec = RANSDecoder(rans_data)

    # Envelope first, then the alpha x activity coded coefficients
    exp_indices = _get_exponents(rans_dec, n_channels, ms_flags)

    active = BAND_EDGES[-1]
    all_quants = [[] for _ in range(n_channels)]
    for ch in range(n_channels):
        # Flat array for activity computation (populated as we decode)
        q_flat = np.zeros(active, dtype=np.int32)

        for b in range(N_BANDS):
            bw_len = BAND_EDGES[b + 1] - BAND_EDGES[b]
            q = np.zeros(bw_len, dtype=np.int32)
            act = _prev_band_activity(q_flat, b)
            ashift = MS_RANS_ALPHA_SHIFT if (ch == 1 and ms_flags is not None and ms_flags[b]) else 0
            tidx = rans_table_idx(b, gain, act, ashift)
            fd, cf, _ = get_rans_tables(tidx)
            for i in range(bw_len):
                sym = rans_dec.get(fd, cf)
                mag = sym
                if sym >= RANS_MAX_SYM - 1:
                    nbits = 0
                    while rans_dec.get(_SIGN_FREQ, _SIGN_CF) == 0:
                        nbits += 1
                    val = 1
                    for _ in range(nbits):
                        val = (val << 1) | rans_dec.get(_SIGN_FREQ, _SIGN_CF)
                    mag = (RANS_MAX_SYM - 1) + val - 1
                if mag > 0:
                    sign = rans_dec.get(_SIGN_FREQ, _SIGN_CF)
                    q[i] = -mag if sign else mag
            # Update flat array for next band's activity
            s = BAND_EDGES[b]
            q_flat[s:s + bw_len] = q
            all_quants[ch].append(q)

    # Reconstruct reflection coefficients
    tns_ks = []
    for ch in range(n_channels):
        if tns_orders[ch] > 0:
            lar_dq = tns_q_ks[ch].astype(np.float64) * TNS_LAR_MAX / float(half)
            tns_ks.append(np.tanh(lar_dq / 2.0))
        else:
            tns_ks.append(np.zeros(0, dtype=np.float64))

    return gain, tns_orders, tns_ks, exp_indices, all_quants, ms_flags
