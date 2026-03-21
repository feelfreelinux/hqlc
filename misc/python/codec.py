"""Top-level encode and decode functions."""

import argparse
import math
import sys
import wave

import numpy as np

from .bitstream import (
    count_side_bits,
    decode_frame,
    encode_frame,
    probe_frame_bits,
)
from .constants import (
    BAND_EDGES,
    BLOCK_SIZE,
    FRAME_LEN,
    FS,
    N_BANDS,
    N_BINS,
)
from .mdct import imdct_synthesis, mdct_analysis
from .psy import (
    NF_MUL,
    NF_SEED_BIAS,
    compute_exponents,
    compute_spreading_envelope,
    nf_decide,
    nf_synthesize,
)
from .quantizer import (
    GAIN_Q,
    GAIN_RC_MAX,
    dequantize_band,
    dequantize_gain,
    quantize_band,
    quantize_gain,
)
from .tns import analyze as tns_analyze
from .tns import lattice_fir, lattice_iir


def _pad_channels(channels):
    """Pad channel arrays for MDCT overlap."""
    N = FRAME_LEN
    n_ch = len(channels)
    ns = min(len(ch) for ch in channels)
    pad_end = (N - ns % N) % N + N
    padded = []
    for ch in channels:
        p = np.zeros(N + ns + pad_end)
        p[N : N + ns] = ch[:ns]
        padded.append(p)
    n_frames = (len(padded[0]) - BLOCK_SIZE) // N + 1
    return padded, n_frames, ns


def encode(channels, gain):
    """Fixed-gain encode. Returns (payloads, total_bits, n_frames)."""
    N = FRAME_LEN
    n_ch = len(channels)
    padded, n_frames, _ = _pad_channels(channels)
    payloads = []
    total_bits = 0

    for fi in range(n_frames):
        start = fi * N
        gain_dq = dequantize_gain(quantize_gain(gain))
        inv_gain = 1.0 / gain_dq

        # MDCT + TNS per channel
        Xs, tns_orders, tns_q_ks = [], [], []
        for ch in range(n_ch):
            block = padded[ch][start : start + BLOCK_SIZE]
            X = mdct_analysis(block)
            order, k_dq, q_k, _ = tns_analyze(X, block)
            if order > 0:
                X = lattice_fir(X, k_dq)
            Xs.append(X)
            tns_orders.append(order)
            tns_q_ks.append(q_k)

        # Exponents + spreading envelope + NF decisions
        exp_indices = [compute_exponents(Xs[ch]) for ch in range(n_ch)]
        smr_q4 = []
        for ch in range(n_ch):
            e_q4, m_q4 = compute_spreading_envelope(exp_indices[ch])
            smr_q4.append(e_q4 - m_q4)
        nf_masks = [
            nf_decide(exp_indices[ch], Xs[ch], smr_q4[ch]) for ch in range(n_ch)
        ]

        # Quantize
        all_quants = []
        for ch in range(n_ch):
            quants = []
            for b in range(N_BANDS):
                s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
                if nf_masks[ch][b]:
                    quants.append(np.zeros(e - s, dtype=np.int32))
                else:
                    quants.append(
                        quantize_band(Xs[ch][s:e], int(exp_indices[ch][b]), inv_gain, b)
                    )
            all_quants.append(quants)

        frame_bits, payload = encode_frame(
            gain, tns_orders, tns_q_ks, exp_indices, nf_masks, all_quants
        )
        payloads.append(payload)
        total_bits += frame_bits

    return payloads, total_bits, n_frames


def decode(payloads, n_channels, n_samples):
    """Decode frame payloads, returns list of channel arrays."""
    N = FRAME_LEN
    pad_end = (N - n_samples % N) % N + N
    total_len = N + n_samples + pad_end
    outputs = [np.zeros(total_len) for _ in range(n_channels)]

    for fi, payload in enumerate(payloads):
        start = fi * N
        gain, tns_orders, tns_ks, exp_indices, nf_masks, all_quants = decode_frame(
            payload, n_channels
        )
        inv_gain = 1.0 / gain

        for ch in range(n_channels):
            X_hat = np.zeros(N_BINS)
            for b in range(N_BANDS):
                s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
                idx = int(exp_indices[ch][b])
                if nf_masks[ch][b]:
                    nf_amp = (2.0 ** ((idx - 43) / 4.0)) * NF_MUL
                    seed = (
                        fi * 2246822519 + (b + ch * N_BANDS) * 3266489917 + NF_SEED_BIAS
                    ) & 0xFFFFFFFF
                    X_hat[s:e] = nf_synthesize(seed, e - s, nf_amp)
                else:
                    X_hat[s:e] = dequantize_band(all_quants[ch][b], idx, inv_gain, b)

            if tns_orders[ch] > 0:
                X_hat = lattice_iir(X_hat, tns_ks[ch])
            outputs[ch][start : start + BLOCK_SIZE] += imdct_synthesis(X_hat)

    return [np.clip(out[N : N + n_samples], -1.0, 1.0) for out in outputs]


def encode_rc(channels, bitrate):
    """Rate-controlled encode with 2-probe gain search.

    Returns (payloads, total_bits, n_frames).
    """
    N = FRAME_LEN
    n_ch = len(channels)
    padded, n_frames, _ = _pad_channels(channels)

    target_bpf = int(bitrate * N / FS)
    tol = max(8, target_bpf // 50)

    # State
    prev_gc = quantize_gain(0.5)
    ema_gc = float(prev_gc)
    prev_side_bits = 150
    res_bits = 0
    payloads = []
    total_bits = 0

    for fi in range(n_frames):
        start = fi * N

        # MDCT + TNS
        Xs, tns_orders, tns_q_ks = [], [], []
        for ch in range(n_ch):
            block = padded[ch][start : start + BLOCK_SIZE]
            X = mdct_analysis(block)
            order, k_dq, q_k, _ = tns_analyze(X, block)
            if order > 0:
                X = lattice_fir(X, k_dq)
            Xs.append(X)
            tns_orders.append(order)
            tns_q_ks.append(q_k)

        # Exponents + NF
        exp_indices = [compute_exponents(Xs[ch]) for ch in range(n_ch)]
        smr_q4 = []
        for ch in range(n_ch):
            e_q4, m_q4 = compute_spreading_envelope(exp_indices[ch])
            smr_q4.append(e_q4 - m_q4)
        nf_masks = [
            nf_decide(exp_indices[ch], Xs[ch], smr_q4[ch]) for ch in range(n_ch)
        ]

        # Rate control: 2-probe gain search
        quiet_frame = False
        borrow = max(-target_bpf, min(target_bpf, res_bits)) // 2
        effective_target = max(
            target_bpf // 4, min(target_bpf * 3, target_bpf + borrow)
        )

        gc0 = min(prev_gc, GAIN_RC_MAX)
        gd0 = dequantize_gain(gc0)
        b0 = probe_frame_bits(Xs, exp_indices, nf_masks, gd0, n_ch) + prev_side_bits

        if abs(b0 - effective_target) <= tol or b0 <= 0:
            chosen_code = gc0
        else:
            ratio = effective_target / max(b0, 1)
            delta = round(float(GAIN_Q) * math.log2(max(ratio, 0.01)))
            oct_above = max(0.0, (prev_gc - ema_gc) / GAIN_Q)
            if oct_above > 1.5:
                slew_dn = GAIN_Q * 3
            elif oct_above > 0.5:
                slew_dn = GAIN_Q * 2
            else:
                slew_dn = GAIN_Q
            delta = max(-slew_dn, min(GAIN_Q, delta))
            if delta == 0:
                delta = 1 if b0 < effective_target else -1

            gc1 = max(0, min(GAIN_RC_MAX, gc0 + delta))
            gd1 = dequantize_gain(gc1)
            b1 = probe_frame_bits(Xs, exp_indices, nf_masks, gd1, n_ch) + prev_side_bits

            if (
                gc1 > gc0
                and b0 < effective_target
                and (b1 - b0) < tol * (gc1 - gc0) // 2
            ):
                chosen_code = gc0
                quiet_frame = True
            elif abs(b1 - effective_target) < abs(b0 - effective_target):
                chosen_code = gc1
            else:
                chosen_code = gc0

        # Quantize at chosen gain
        gain_dq = dequantize_gain(chosen_code)
        inv_gain = 1.0 / gain_dq
        all_quants = []
        for ch in range(n_ch):
            quants = []
            for b in range(N_BANDS):
                s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
                if nf_masks[ch][b]:
                    quants.append(np.zeros(e - s, dtype=np.int32))
                else:
                    quants.append(
                        quantize_band(Xs[ch][s:e], int(exp_indices[ch][b]), inv_gain, b)
                    )
            all_quants.append(quants)

        frame_bits, payload = encode_frame(
            gain_dq, tns_orders, tns_q_ks, exp_indices, nf_masks, all_quants
        )
        payloads.append(payload)
        total_bits += frame_bits

        # Update RC state
        if not quiet_frame:
            res_bits += target_bpf - frame_bits
            res_bits = max(-(2 * target_bpf), min(2 * target_bpf, res_bits))
            ema_gc += (chosen_code - ema_gc) / 16.0
        prev_gc = chosen_code
        prev_side_bits = (
            count_side_bits(n_ch, tns_orders, tns_q_ks, exp_indices, nf_masks) + 32
        )

    return payloads, total_bits, n_frames


def run(audio_l, audio_r, gain):
    """Fixed-gain encode + decode stereo."""
    payloads, total_bits, n_frames = encode([audio_l, audio_r], gain)
    ns = min(len(audio_l), len(audio_r))
    dec_l, dec_r = decode(payloads, 2, ns)
    return dec_l, dec_r, total_bits, n_frames


def run_rc(audio_l, audio_r, bitrate):
    """Rate-controlled encode + decode stereo."""
    payloads, total_bits, n_frames = encode_rc([audio_l, audio_r], bitrate)
    ns = min(len(audio_l), len(audio_r))
    dec_l, dec_r = decode(payloads, 2, ns)
    return dec_l, dec_r, total_bits, n_frames


def _wav_read(path):
    """Read a 16-bit PCM WAV and return (channels_list, sample_rate)."""
    with wave.open(path, "rb") as wf:
        n_ch = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    samples = samples.reshape(-1, n_ch)
    channels = [samples[:, ch] for ch in range(n_ch)]
    return channels, sr


def _wav_write(path, channels, sample_rate):
    """Write channels as 16-bit PCM WAV."""
    n_ch = len(channels)
    n_frames = len(channels[0])
    interleaved = np.column_stack(channels)
    pcm = np.clip(interleaved * 32768.0, -32768, 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_ch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="hqlc",
        description="HQLC Python reference encoder/decoder",
    )
    parser.add_argument("input", help="input WAV file (16-bit PCM, 48 kHz)")
    parser.add_argument("output", help="output WAV file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--bitrate",
        type=int,
        default=128000,
        help="rate-controlled mode at given bitrate (default: 128000)",
    )
    group.add_argument(
        "-g", "--gain", type=float, default=None, help="fixed-gain mode (e.g. 2.0)"
    )
    args = parser.parse_args()

    channels, sr = _wav_read(args.input)
    if sr != FS:
        print(f"error: sample rate must be {FS} (got {sr})", file=sys.stderr)
        sys.exit(1)

    n_ch = len(channels)
    ns = min(len(ch) for ch in channels)

    if args.gain is not None:
        payloads, total_bits, n_frames = encode(channels, args.gain)
        mode_str = f"fixed (gain {args.gain:.2f})"
    else:
        payloads, total_bits, n_frames = encode_rc(channels, args.bitrate)
        mode_str = f"RC (target {args.bitrate} bps)"

    decoded = decode(payloads, n_ch, ns)

    # trim 1-frame latency
    for i in range(n_ch):
        decoded[i] = decoded[i][FRAME_LEN:]
    out_samples = len(decoded[0])

    _wav_write(args.output, decoded, sr)

    duration = out_samples / sr
    avg_bitrate = total_bits / duration
    input_bitrate = sr * n_ch * 16
    ratio = input_bitrate / avg_bitrate

    print(f"{args.input} -> {args.output}")
    print(f"  {n_frames} frames, {duration:.2f}s, {n_ch}ch")
    print(f"  mode: {mode_str}")
    print(f"  avg bitrate: {avg_bitrate:.0f} bps ({ratio:.1f}:1)")


if __name__ == "__main__":
    main()
