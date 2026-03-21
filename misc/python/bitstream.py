"""Frame serialization: side info encoding/decoding and bit counting.

Frame layout:
  [gain_code: 7b] [TNS per ch] [exponents] [NF masks] [padding] [rANS payload]

TNS per channel:
  [active: 1b] [order-1: 3b if active] [LAR indices: 4b each if active]

Exponents:
  Ch0: Rice-coded DPCM (3-bit k + zigzag + Rice codes)
  Ch1+: Rice-coded delta from ch0

NF masks: 1 bit per band per channel
Padding: 0-7 bits to byte-align before rANS payload
"""

import numpy as np

from .constants import BAND_EDGES, N_BANDS
from .entropy import (
    _SIGN_CF,
    _SIGN_FREQ,
    RANS_MAX_SYM,
    BitReader,
    BitWriter,
    RANSDecoder,
    RANSEncoder,
    build_band_tables,
    coeff_cost_q8,
    find_best_rice_k,
    zigzag_dec,
    zigzag_enc,
)
from .quantizer import GAIN_BITS, dequantize_gain, quantize_gain
from .tns import TNS_K_BITS, TNS_LAR_MAX


def count_side_bits(n_ch, tns_orders, tns_q_ks, exp_indices, nf_masks):
    """Count side information bits (gain + TNS + exponents + NF + padding)."""
    bits = GAIN_BITS
    for ch in range(n_ch):
        bits += 1
        if tns_orders[ch] > 0:
            bits += 3 + TNS_K_BITS * tns_orders[ch]

    # Ch0 exponent DPCM
    deltas = []
    prev = 0
    for idx in exp_indices[0]:
        deltas.append(int(idx) - prev)
        prev = int(idx)
    k = find_best_rice_k(deltas)
    bits += 3 + sum((zigzag_enc(d) >> k) + 1 + k for d in deltas)

    # Ch1+ delta from ch0
    for ch in range(1, n_ch):
        deltas = [
            int(exp_indices[ch][b]) - int(exp_indices[0][b]) for b in range(N_BANDS)
        ]
        k = find_best_rice_k(deltas)
        bits += 3 + sum((zigzag_enc(d) >> k) + 1 + k for d in deltas)

    bits += N_BANDS * n_ch  # NF masks
    bits += (8 - bits % 8) % 8  # byte alignment
    return bits


def probe_frame_bits(Xs, exp_indices, nf_masks, gain_dq, n_ch):
    """Estimate payload bits at a given gain (fast, no actual rANS stream)."""
    from .constants import BAND_WEIGHTS, DEAD_ZONE, EXP_INDEX_BIAS

    _, _, cost_tables = build_band_tables(gain_dq)
    inv_gain = 1.0 / gain_dq
    total_q8 = 0

    for ch in range(n_ch):
        for b in range(N_BANDS):
            if nf_masks[ch][b]:
                continue
            s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
            idx = int(exp_indices[ch][b])
            step = (2.0 ** ((idx - EXP_INDEX_BIAS) / 4.0)) * BAND_WEIGHTS[b] * inv_gain
            scaled = np.abs(Xs[ch][s:e]) / step
            mags = np.where(
                scaled > DEAD_ZONE,
                np.floor(scaled - DEAD_ZONE + 1.0).astype(np.int32),
                0,
            )
            for mag in mags:
                total_q8 += coeff_cost_q8(cost_tables[b], int(mag))

    return (total_q8 + 128) >> 8


def encode_frame(gain, tns_orders, tns_q_ks, exp_indices, nf_masks, all_quants):
    """Encode one frame, returns (total_bits, payload_bytes)."""
    n_ch = len(tns_orders)
    side = BitWriter()

    gain_code = quantize_gain(gain)
    side.write(gain_code, GAIN_BITS)

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

    # Ch0 exponents: DPCM
    deltas = []
    prev = 0
    for idx in exp_indices[0]:
        deltas.append(int(idx) - prev)
        prev = int(idx)
    k = find_best_rice_k(deltas)
    side.write(k, 3)
    for d in deltas:
        side.write_rice(zigzag_enc(d), k)

    # Ch1+ exponents: delta from ch0
    for ch in range(1, n_ch):
        ch_deltas = [
            int(exp_indices[ch][b]) - int(exp_indices[0][b]) for b in range(N_BANDS)
        ]
        k = find_best_rice_k(ch_deltas)
        side.write(k, 3)
        for d in ch_deltas:
            side.write_rice(zigzag_enc(d), k)

    # NF masks
    for ch in range(n_ch):
        for b in range(N_BANDS):
            side.write(1 if nf_masks[ch][b] else 0, 1)

    # Byte align
    pad = (8 - side.total_bits() % 8) % 8
    if pad:
        side.write(0, pad)
    side_bytes = side.get_bytes()

    # rANS coefficients
    gain_dq = dequantize_gain(gain_code)
    freq_lists, cf_lists, _ = build_band_tables(gain_dq)

    rans_enc = RANSEncoder()
    for ch in range(n_ch):
        for b in range(N_BANDS):
            if nf_masks[ch][b]:
                continue
            q = all_quants[ch][b]
            fd, cf = freq_lists[b], cf_lists[b]
            for v in q:
                mag = abs(int(v))
                rans_enc.put(min(mag, RANS_MAX_SYM - 1), fd, cf)
                if mag >= RANS_MAX_SYM - 1:
                    overflow = mag - (RANS_MAX_SYM - 1)
                    nbits = (overflow + 1).bit_length() - 1
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


def decode_frame(payload, n_channels):
    """Decode one frame from payload bytes.

    Returns (gain, tns_orders, tns_ks, exp_indices, nf_masks, all_quants).
    """
    br = BitReader(payload)
    gain_code = br.read(GAIN_BITS)
    gain = dequantize_gain(gain_code)

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

    # Exponents
    exp_indices = [np.zeros(N_BANDS, dtype=np.int32) for _ in range(n_channels)]
    k0 = br.read(3)
    prev = 0
    for b in range(N_BANDS):
        u = br.read_rice(k0)
        exp_indices[0][b] = prev + zigzag_dec(u)
        prev = exp_indices[0][b]
    for ch in range(1, n_channels):
        k = br.read(3)
        for b in range(N_BANDS):
            exp_indices[ch][b] = exp_indices[0][b] + zigzag_dec(br.read_rice(k))

    # NF masks
    nf_masks = [np.zeros(N_BANDS, dtype=bool) for _ in range(n_channels)]
    for ch in range(n_channels):
        for b in range(N_BANDS):
            nf_masks[ch][b] = br.read(1) == 1

    # Byte align to rANS start
    bits_used = br.bits_read()
    pad = (8 - bits_used % 8) % 8
    if pad:
        br.read(pad)
    rans_start = br.bits_read() // 8

    # rANS decode
    rans_data = payload[rans_start:]
    freq_lists, cf_lists, _ = build_band_tables(gain)
    rans_dec = RANSDecoder(rans_data) if len(rans_data) > 0 else None

    all_quants = [[] for _ in range(n_channels)]
    for ch in range(n_channels):
        for b in range(N_BANDS):
            bw_len = BAND_EDGES[b + 1] - BAND_EDGES[b]
            if nf_masks[ch][b]:
                all_quants[ch].append(np.zeros(bw_len, dtype=np.int32))
                continue
            q = np.zeros(bw_len, dtype=np.int32)
            fd, cf = freq_lists[b], cf_lists[b]
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
            all_quants[ch].append(q)

    # Reconstruct reflection coefficients
    tns_ks = []
    for ch in range(n_channels):
        if tns_orders[ch] > 0:
            lar_dq = tns_q_ks[ch].astype(np.float64) * TNS_LAR_MAX / float(half)
            tns_ks.append(np.tanh(lar_dq / 2.0))
        else:
            tns_ks.append(np.zeros(0, dtype=np.float64))

    return gain, tns_orders, tns_ks, exp_indices, nf_masks, all_quants
