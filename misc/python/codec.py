"""Top-level encoder / decoder pipeline. Read the HQLC_DESIGN.md for details.

Encode, per frame of 512 new samples:
    MDCT -> transient-gated TNS -> M/S when eligible -> band exponents (psy) -> quantize -> rANS.

Decide just reverses it:
    rANS -> dequantize + noise fill -> M/S back into L/R -> TNS synthesis -> inverse MDCT + overlap-add.
"""

import argparse
import math
import sys
import wave

import numpy as np

from .bitstream import (
    count_side_bits,
    decode_frame,
    encode_frame,
)
from .constants import (
    BAND_EDGES,
    BLOCK_SIZE,
    DEAD_ZONE,
    FRAME_LEN,
    FS,
    N_BANDS,
    N_BINS,
)
from .entropy import (
    _prev_band_activity,
    coeff_cost_q8,
    get_rans_tables,
    rans_table_idx,
)
from .mdct import imdct_synthesis, mdct_analysis
from .ms import ms_apply_side_exp_bias, ms_decode, ms_encode
from .psy import compute_exponents, tilt_for_bitrate
from .quantizer import (
    GAIN_Q,
    GAIN_RC_MAX,
    dequantize,
    dequantize_gain,
    exp_to_step_factor,
    quantize,
    quantize_gain,
)
from .tns import TransientDetector, lattice_fir, lattice_iir
from .tns import analyze as tns_analyze


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


def _band_step_factors(exp_indices):
    """Flat per-band step factor per bin

    Used on TNS frames, for the RC probe, and for the NF fill amplitude.
    The caller can get the actual step factor by dividing this by the global gain.
    """
    active = BAND_EDGES[-1]
    env = np.zeros(active)
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        env[s:e] = exp_to_step_factor(int(exp_indices[b]))
    return env


def _interp_bin_exp(exp_indices):
    """Per-bin exponent indices, linearly interpolated between band centers"""
    active = BAND_EDGES[-1]
    centers = [(BAND_EDGES[b] + BAND_EDGES[b + 1] + 2) >> 1 for b in range(N_BANDS)]
    bin_exp = np.zeros(active, dtype=np.int32)
    prev = centers[0]
    bin_exp[:prev] = exp_indices[0]
    for b in range(N_BANDS - 1):
        nxt = centers[b + 1]
        v0 = float(exp_indices[b])
        dv = float(exp_indices[b + 1]) - v0
        dist = float(nxt - prev)
        for i in range(prev, nxt):
            bin_exp[i] = int(math.floor(v0 + dv * (i - prev + 0.5) / dist + 0.5))
        prev = nxt
    bin_exp[prev:active] = exp_indices[N_BANDS - 1]
    return bin_exp


def _step_factors(exp_indices, tns_order):
    """Returns the per-bin step factors for given exponent indices.
    For transient / TNS frames, the factors are flat per-band.
    For steady frames, we use the interpolated bin exponents."""
    if tns_order == 0:
        return exp_to_step_factor(_interp_bin_exp(exp_indices).astype(np.float64))
    return _band_step_factors(exp_indices)


def _envelope_quantize(Xs, envs, gain):
    """Deadzone-quantize each channel, split the symbols into bands.

    Returns (all_quants, all_scaled_abs) all_scaled_abs[ch] is the pre-deadzone |X * gain / step| that the NF estimator uses
    """
    inv_gain = 1.0 / gain
    all_quants = []
    all_scaled_abs = []
    for ch in range(len(Xs)):
        q, abs_scaled = quantize(Xs[ch], envs[ch] * inv_gain)
        all_quants.append(
            [q[BAND_EDGES[b] : BAND_EDGES[b + 1]].copy() for b in range(N_BANDS)]
        )
        all_scaled_abs.append(abs_scaled)
    return all_quants, all_scaled_abs


def _estimate_noise_factor(scaled_abs, active):
    """Estimate the noise fill factor from pre-deadzone |scaled| values.

    Averages |X|/step over the bins the decoder will fill (runs of >4 sub-deadzone bins, 2-bin lag).
    Returns nf in 0..7, 0 = strongest fill.
    """
    z = 0
    total = 0.0
    ns = 0

    for i in range(active):
        if scaled_abs[i] < DEAD_ZONE:
            z += 1
        else:
            z = 0
        if z > 4:
            total += scaled_abs[i - 2]
            ns += 1

    if ns == 0:
        return 7

    # matches C
    nf = int(math.floor(8.5 - 16.0 * (total / ns)))
    return max(0, min(7, nf))


def _dequant_and_fill(q, gain, nf, env_dq, env_flat, seed, active, skip_bins=None):
    """Dequantize, then fill noise into the run of quantized zeros.

    skip_bins (bool per bin) skips NF in given bands, use for S channel in M/S
    """
    inv_gain = 1.0 / gain
    X_hat = np.zeros(N_BINS)
    X_hat[:active] = dequantize(q, env_dq * inv_gain)

    nf_amp = (8 - nf) / (16.0 * gain)
    z = 0
    for i in range(active):
        z = z + 1 if q[i] == 0 else 0
        if z > 4:
            # LCG seed
            seed = (13849 + seed * 31821) & 0xFFFF
            if skip_bins is not None and skip_bins[i - 2]:
                continue  # no noise fill on flagged side bands
            sign = -1.0 if (seed & 0x8000) else 1.0
            X_hat[i - 2] = sign * nf_amp * env_flat[i]

    return X_hat


def _probe_bits(Xs, envs, gain, n_ch, ms_flags=None):
    """Estimate rANS payload bits at a candidate gain (flat per-band steps).

    Sums the cost of every quantized coefficient without actually entropy-coding it
    """
    from .ms import MS_RANS_ALPHA_SHIFT

    total_q8 = 0
    inv_gain = 1.0 / gain

    for ch in range(n_ch):
        q, _ = quantize(Xs[ch], envs[ch] * inv_gain)  # flat per-bin symbols
        for b in range(N_BANDS):
            s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
            act = _prev_band_activity(q, b)
            ashift = (
                MS_RANS_ALPHA_SHIFT
                if (ch == 1 and ms_flags is not None and ms_flags[b])
                else 0
            )
            _, _, cost_q8 = get_rans_tables(rans_table_idx(b, gain, act, ashift))
            total_q8 += sum(coeff_cost_q8(cost_q8, int(v)) for v in q[s:e])

    return ((total_q8 + 128) >> 8) + 32  # +32 for rANS state flush


def _analyze_frame(padded, start, n_ch, tilt_db, detectors, tns_hang, ms_flags=None):
    """MDCT + transient-gated TNS + exponents for one frame"""
    N = FRAME_LEN
    # MDCT + transient detection per channel
    raw, eligible = [], []
    for ch in range(n_ch):
        block = padded[ch][start : start + BLOCK_SIZE]
        raw.append(mdct_analysis(block))
        # TNS runs when a transient fires this frame or the hangover is still up
        transient = detectors[ch].detect(block[N:])
        eligible.append(transient or tns_hang[ch] > 0)
        tns_hang[ch] = 1 if transient else max(0, tns_hang[ch] - 1)

    # Apply M/S stereo coding only on non-transient and eligible channels
    if ms_flags is not None:
        if eligible[0] or eligible[1]:
            ms_flags[:] = False
        else:
            ms_encode(raw[0], raw[1], ms_flags)

    # Transient-gated TNS
    Xs, tns_orders, tns_q_ks = [], [], []
    for ch in range(n_ch):
        X = raw[ch]
        order, q_k = 0, np.zeros(0, dtype=np.int32)
        if eligible[ch]:
            order, k_dq, q_k, _ = tns_analyze(X)
            if order > 0:
                X = lattice_fir(X, k_dq)
        Xs.append(X)
        tns_orders.append(order)
        tns_q_ks.append(q_k)

    exp_indices = [compute_exponents(Xs[ch], tilt_db, eligible[ch]) for ch in range(n_ch)]

    # Coarsen the flagged side-channel exponents
    if ms_flags is not None:
        ms_apply_side_exp_bias(exp_indices[0], exp_indices[1], ms_flags)

    return Xs, tns_orders, tns_q_ks, exp_indices


def encode(channels, gain):
    """Fixed-gain encode. Returns (payloads, total_bits, n_frames)."""
    N = FRAME_LEN
    n_ch = len(channels)
    padded, n_frames, _ = _pad_channels(channels)
    active = BAND_EDGES[-1]
    tilt_db = tilt_for_bitrate(
        96000
    )  # default tilt for fixed-gain mode, probably should scale with bitrate
    detectors = [TransientDetector() for _ in range(n_ch)]
    tns_hang = [0] * n_ch
    use_ms = n_ch == 2
    ms_flags = np.zeros(N_BANDS, dtype=bool) if use_ms else None
    payloads = []
    total_bits = 0

    for fi in range(n_frames):
        start = fi * N
        Xs, tns_orders, tns_q_ks, exp_indices = _analyze_frame(
            padded, start, n_ch, tilt_db, detectors, tns_hang, ms_flags
        )

        envs = [_step_factors(exp_indices[ch], tns_orders[ch]) for ch in range(n_ch)]
        all_quants, all_scaled_abs = _envelope_quantize(Xs, envs, gain)

        noise_factors = [
            _estimate_noise_factor(all_scaled_abs[ch], active) for ch in range(n_ch)
        ]

        frame_bits, payload = encode_frame(
            gain,
            tns_orders,
            tns_q_ks,
            exp_indices,
            all_quants,
            noise_factors,
            ms_flags if use_ms else None,
        )
        payloads.append(payload)
        total_bits += frame_bits

    return payloads, total_bits, n_frames


def decode(payloads, n_channels, n_samples):
    """Decode frame payloads: dequant + NF fill, then TNS synthesis + IMDCT."""
    N = FRAME_LEN
    pad_end = (N - n_samples % N) % N + N
    total_len = N + n_samples + pad_end
    outputs = [np.zeros(total_len) for _ in range(n_channels)]

    active = BAND_EDGES[-1]
    use_ms = n_channels == 2
    for fi, payload in enumerate(payloads):
        start = fi * N
        gain, tns_orders, tns_ks, exp_indices, all_quants, noise_factors, ms_flags = (
            decode_frame(payload, n_channels, use_ms)
        )

        # Calculate the skip bins from ms mask
        skip_bins = None
        if use_ms:
            skip_bins = np.zeros(active, dtype=bool)
            for b in range(N_BANDS):
                if ms_flags[b]:
                    skip_bins[BAND_EDGES[b] : BAND_EDGES[b + 1]] = True

        # dequant + nf + TNS synthesis per channel
        specs = []
        for ch in range(n_channels):
            env_flat = _band_step_factors(exp_indices[ch])
            env_dq = _step_factors(exp_indices[ch], tns_orders[ch])

            q_flat = np.zeros(active, dtype=np.int32)
            for b in range(N_BANDS):
                s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
                q_flat[s:e] = all_quants[ch][b]

            # Seed varies per frame and channel so the fill noise evolves
            seed = (0x9E3779B9 ^ (fi * 0x9E37) ^ (ch * 0x51ED)) & 0xFFFFFFFF
            X_hat = _dequant_and_fill(
                q_flat,
                gain,
                noise_factors[ch],
                env_dq,
                env_flat,
                seed,
                active,
                skip_bins if ch == 1 else None, # Skip NF on S channel
            )

            if tns_orders[ch] > 0:
                X_hat = lattice_iir(X_hat, tns_ks[ch])
            specs.append(X_hat)

        if use_ms:
            # Decode the M/S back into L/R
            ms_decode(specs[0], specs[1], ms_flags)
        for ch in range(n_channels):
            outputs[ch][start : start + BLOCK_SIZE] += imdct_synthesis(specs[ch])

    return [np.clip(out[N : N + n_samples], -1.0, 1.0) for out in outputs]


def _slew_limit(prev_gc, ema_gc):
    """Max downward gain-code step from distance above the long-term EMA.

    Matches C compute_slew_limit: >1.5 oct -> 3*Q, >0.5 oct -> 2*Q, else Q
    (8 gain codes per octave, so 1.5 oct = 12 codes, 0.5 oct = 4 codes).
    """
    oct_above = max(0.0, prev_gc - ema_gc)
    if oct_above > 12:
        return GAIN_Q * 3
    if oct_above > 4:
        return GAIN_Q * 2
    return GAIN_Q


def encode_rc(channels, bitrate):
    """Rate-controlled encode. Returns (payloads, total_bits, n_frames)."""
    N = FRAME_LEN
    n_ch = len(channels)
    padded, n_frames, _ = _pad_channels(channels)
    active = BAND_EDGES[-1]
    tilt_db = tilt_for_bitrate(bitrate)

    target_bpf = int(bitrate * N / FS)
    tol = max(8, target_bpf // 50)

    prev_gc = quantize_gain(16.0)
    ema_gc = float(prev_gc)
    prev_side_bits = 150
    res_bits = 0
    detectors = [TransientDetector() for _ in range(n_ch)]
    tns_hang = [0] * n_ch
    use_ms = n_ch == 2
    ms_flags = np.zeros(N_BANDS, dtype=bool) if use_ms else None
    payloads = []
    total_bits = 0

    for fi in range(n_frames):
        start = fi * N
        Xs, tns_orders, tns_q_ks, exp_indices = _analyze_frame(
            padded, start, n_ch, tilt_db, detectors, tns_hang, ms_flags
        )

        # Probes always use flat per-band steps
        envs_flat = [_band_step_factors(exp_indices[ch]) for ch in range(n_ch)]

        # Rate control: slew-limited 2-probe gain search
        quiet_frame = False
        borrow = max(-target_bpf, min(target_bpf, res_bits)) // 2
        effective_target = max(
            target_bpf // 4, min(target_bpf * 3, target_bpf + borrow)
        )

        gc0 = min(prev_gc, GAIN_RC_MAX)
        b0 = (
            _probe_bits(Xs, envs_flat, dequantize_gain(gc0), n_ch, ms_flags)
            + prev_side_bits
        )

        if abs(b0 - effective_target) <= tol or b0 <= 0:
            chosen_code = gc0
        else:
            # estimate_gain_delta: round(Q * log2(target / probed))
            delta = round(float(GAIN_Q) * math.log2(effective_target / max(b0, 1)))
            slew_dn = _slew_limit(prev_gc, ema_gc)
            delta = max(-slew_dn, min(GAIN_Q, delta))
            if delta == 0:
                delta = 1 if b0 < effective_target else -1

            gc1 = max(0, min(GAIN_RC_MAX, gc0 + delta))
            b1 = (
                _probe_bits(Xs, envs_flat, dequantize_gain(gc1), n_ch, ms_flags)
                + prev_side_bits
            )

            if (
                gc1 > gc0
                and b0 < effective_target
                and (b1 - b0) < tol * (gc1 - gc0) // 2
            ):
                chosen_code = gc0
                quiet_frame = True
            else:
                chosen_code = (
                    gc1
                    if abs(b1 - effective_target) < abs(b0 - effective_target)
                    else gc0
                )

        gain = dequantize_gain(chosen_code)
        envs = [_step_factors(exp_indices[ch], tns_orders[ch]) for ch in range(n_ch)]
        all_quants, all_scaled_abs = _envelope_quantize(Xs, envs, gain)

        noise_factors = [
            _estimate_noise_factor(all_scaled_abs[ch], active) for ch in range(n_ch)
        ]

        frame_bits, payload = encode_frame(
            gain,
            tns_orders,
            tns_q_ks,
            exp_indices,
            all_quants,
            noise_factors,
            ms_flags if use_ms else None,
        )
        payloads.append(payload)
        total_bits += frame_bits

        if not quiet_frame:
            res_bits += target_bpf - frame_bits
            res_bits = max(-(2 * target_bpf), min(2 * target_bpf, res_bits))
            ema_gc += (chosen_code - ema_gc) / 16.0
        prev_gc = chosen_code
        prev_side_bits = (
            count_side_bits(
                n_ch, tns_orders, tns_q_ks, exp_indices, ms_flags if use_ms else None
            )
            + 32
        )

    return payloads, total_bits, n_frames


# WAV I/O
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


# CLI
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
        "-g", "--gain", type=float, default=None, help="fixed-gain mode (e.g. 16.0)"
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

    _wav_write(args.output, decoded, sr)

    duration = ns / sr
    avg_bitrate = total_bits / duration
    input_bitrate = sr * n_ch * 16
    ratio = input_bitrate / avg_bitrate

    print(f"{args.input} -> {args.output}")
    print(f"  {n_frames} frames, {duration:.2f}s, {n_ch}ch")
    print(f"  mode: {mode_str}")
    print(f"  avg bitrate: {avg_bitrate:.0f} bps ({ratio:.1f}:1)")


if __name__ == "__main__":
    main()
