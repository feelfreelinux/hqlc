#!/usr/bin/env python3
"""Experiment: TNS gate tuning + TNS vs TES (Temporal Energy Shaping).

Tests:
  1. TNS on (current: energy gate + spectral LP)
  2. TNS off (baseline)
  3. TNS no-gate (prediction gain only, no energy ratio check)
  4. TES (pre-MDCT PCM attenuation on transient regions, undone after OLA)

TES applies gain reduction to the loud portion of the PCM *before* any MDCT
processing. Since both overlapping frames see the same modified PCM, OLA
reconstruction stays valid. After OLA decode, the inverse gain restores levels.
Quantization noise generated in the attenuated domain gets amplified only in
the loud region where it's perceptually masked.

Usage:
    python scripts/experiment_tns_vs_tes.py test-clips/castanets48_stereo.wav
    python scripts/experiment_tns_vs_tes.py --all
"""

import sys
import os
import wave
import math
import subprocess

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "misc"))

from python import mdct, tns, psy, quantizer
from python.constants import FRAME_LEN, BLOCK_SIZE, N_BANDS, BAND_EDGES, FS

ZIM = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "zimtohrli", "build", "compare")


def read_wav_mono(path):
    with wave.open(path, "r") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        ch = w.getnchannels()
        return data[0::ch].astype(np.float64) / 32768.0


def write_wav_stereo(path, mono):
    with wave.open(path, "w") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(48000)
        s = np.clip(mono * 32768, -32768, 32767).astype(np.int16)
        out = np.empty(len(s) * 2, dtype=np.int16)
        out[0::2] = s
        out[1::2] = s
        w.writeframes(out.tobytes())


def zimtohrli_mos(ref_path, deg_path):
    r = subprocess.run([ZIM, "--path_a", ref_path, "--path_b", deg_path],
                       capture_output=True, text=True)
    for line in r.stdout.strip().splitlines():
        try:
            return float(line)
        except ValueError:
            continue
    return 0.0


def pad_audio(audio):
    N = FRAME_LEN
    ns = len(audio)
    pad_end = (N - ns % N) % N + N
    padded = np.zeros(N + ns + pad_end)
    padded[N:N + ns] = audio
    n_frames = (len(padded) - BLOCK_SIZE) // N + 1
    return padded, n_frames, ns


def encode_decode_core(padded, n_frames, ns, tns_fn):
    """Core encode/decode loop. tns_fn(X, block) -> (order, k_dq) or None."""
    N = FRAME_LEN
    gain_code = quantizer.quantize_gain(1.0)
    inv_gain = 1.0 / quantizer.dequantize_gain(gain_code)
    output = np.zeros(len(padded))

    for fi in range(n_frames):
        start = fi * N
        block = padded[start:start + BLOCK_SIZE]
        X = mdct.mdct_analysis(block)

        order, k_dq = 0, np.zeros(0)
        if tns_fn:
            result = tns_fn(X, block)
            if result is not None:
                order, k_dq = result
                X = tns.lattice_fir(X, k_dq)

        exp_indices = psy.compute_exponents(X)
        e_q4, m_q4 = psy.compute_spreading_envelope(exp_indices)
        nf_mask = psy.nf_decide(exp_indices, X, e_q4 - m_q4)

        X_hat = np.zeros(len(X))
        for b in range(N_BANDS):
            s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
            idx = int(exp_indices[b])
            if nf_mask[b]:
                seed = (fi * 2246822519 + b * 3266489917 + psy.NF_SEED_BIAS) & 0xFFFFFFFF
                X_hat[s:e] = psy.nf_synthesize(
                    seed, e - s, psy.NF_MUL * (2.0 ** ((idx - 43) / 4.0)))
            else:
                q = quantizer.quantize_band(X[s:e], idx, inv_gain, b)
                X_hat[s:e] = quantizer.dequantize_band(q, idx, inv_gain, b)

        if order > 0:
            X_hat = tns.lattice_iir(X_hat, k_dq)

        output[start:start + BLOCK_SIZE] += mdct.imdct_synthesis(X_hat)

    return output[FRAME_LEN:FRAME_LEN + ns]


# ── TNS variants ──

def tns_normal(X, block):
    """Current TNS: energy gate + prediction gain."""
    order, k_dq, _, _ = tns.analyze(X, block)
    return (order, k_dq) if order > 0 else None


def tns_no_gate(X, block):
    """TNS with no energy gate -- prediction gain only."""
    r = tns._autocorrelation(X, tns.TNS_MAX_ORDER)
    k_raw, order, pred_gain = tns._levinson_durbin(r, tns.TNS_MAX_ORDER)
    if order == 0 or pred_gain < tns.TNS_PRED_GAIN_THR:
        return None
    k_raw = np.clip(k_raw, -tns.TNS_MAX_K, tns.TNS_MAX_K)
    q_k, k_dq = tns._quant_lar(k_raw)
    while len(k_dq) > 0 and q_k[-1] == 0:
        q_k, k_dq = q_k[:-1], k_dq[:-1]
    if len(k_dq) == 0:
        return None
    return len(k_dq), k_dq


# ── TES (Temporal Energy Shaping) ──

def apply_tes(audio):
    """Apply TES to raw PCM: attenuate loud transient regions.

    Returns (modified_pcm, gains_per_sample) where gains_per_sample lets the
    decoder undo the modification after OLA.
    """
    N = FRAME_LEN
    ns = len(audio)
    modified = audio.copy()
    gains = np.ones(ns)

    # Detect transients using short-time energy in N-sample hops
    for i in range(N, ns - N, N):
        e_before = np.sum(audio[i - N:i] ** 2) + 1e-30
        e_after = np.sum(audio[i:i + N] ** 2)
        ratio = e_after / e_before

        if ratio >= 6.0:
            # Transient detected at sample i
            # Attenuate the loud region to match the quiet region's RMS
            rms_before = math.sqrt(e_before / N)
            rms_after = math.sqrt(e_after / N) + 1e-15
            # Target: reduce loud half to geometric mean of both
            target = math.sqrt(rms_before * rms_after)
            atten = min(target / rms_after, 1.0)  # only attenuate, never boost

            if atten < 0.95:  # only bother if meaningful attenuation
                # Apply smooth gain ramp over a transition region
                trans = min(64, N // 4)
                # Ramp from 1.0 to atten
                ramp = np.linspace(1.0, atten, trans)
                # Region: ramp down, hold attenuated, ramp back up
                hold_end = min(i + N, ns)
                ramp_up_end = min(hold_end + trans, ns)

                # Ramp down
                end = min(i + trans, ns)
                modified[i:end] *= ramp[:end - i]
                gains[i:end] *= ramp[:end - i]

                # Hold
                modified[end:hold_end] *= atten
                gains[end:hold_end] *= atten

                # Ramp back up
                if hold_end < ramp_up_end:
                    ramp_up = np.linspace(atten, 1.0, ramp_up_end - hold_end)
                    modified[hold_end:ramp_up_end] *= ramp_up
                    gains[hold_end:ramp_up_end] *= ramp_up

    return modified, gains


def encode_decode_tes(audio):
    """Encode with TES: modify PCM before MDCT, undo after OLA."""
    modified, gains = apply_tes(audio)

    # Count TES applications
    tes_count = np.sum(np.abs(np.diff(gains)) > 0.01)

    # Encode/decode the modified PCM (no TNS)
    padded, n_frames, ns = pad_audio(modified)
    decoded_modified = encode_decode_core(padded, n_frames, ns, tns_fn=None)

    # Undo TES gain on decoded output
    decoded = decoded_modified / (gains[:len(decoded_modified)] + 1e-15)

    return decoded, int(tes_count)


# ── Main ──

def run_clip(path):
    name = os.path.basename(path).replace("48_stereo.wav", "").replace(".wav", "")
    audio = read_wav_mono(path)
    print(f"\n{'='*60}")
    print(f"{name} ({len(audio)} samples, {len(audio)/FS:.2f}s)")
    print(f"{'='*60}")

    padded, n_frames, ns = pad_audio(audio)

    # Write reference
    write_wav_stereo(f"/tmp/{name}_ref.wav", audio)

    variants = {}

    # TNS on (current)
    dec = encode_decode_core(padded, n_frames, ns, tns_normal)
    write_wav_stereo(f"/tmp/{name}_tns_on.wav", dec)
    variants["TNS on"] = f"/tmp/{name}_tns_on.wav"

    # TNS off
    dec = encode_decode_core(padded, n_frames, ns, tns_fn=None)
    write_wav_stereo(f"/tmp/{name}_tns_off.wav", dec)
    variants["TNS off"] = f"/tmp/{name}_tns_off.wav"

    # TNS no-gate
    dec = encode_decode_core(padded, n_frames, ns, tns_no_gate)
    write_wav_stereo(f"/tmp/{name}_tns_nogate.wav", dec)
    variants["TNS no-gate"] = f"/tmp/{name}_tns_nogate.wav"

    # TES
    dec, tes_transitions = encode_decode_tes(audio)
    write_wav_stereo(f"/tmp/{name}_tes.wav", dec)
    variants["TES"] = f"/tmp/{name}_tes.wav"
    print(f"  TES: {tes_transitions} gain transitions")

    # Count TNS triggers for each variant
    for label, tns_fn in [("TNS on", tns_normal), ("TNS no-gate", tns_no_gate)]:
        count = 0
        for fi in range(n_frames):
            start = fi * FRAME_LEN
            block = padded[start:start + BLOCK_SIZE]
            X = mdct.mdct_analysis(block)
            if tns_fn(X, block) is not None:
                count += 1
        print(f"  {label}: triggered {count}/{n_frames} frames")

    # Zimtohrli scores
    if os.path.isfile(ZIM):
        ref = f"/tmp/{name}_ref.wav"
        print(f"\n  {'Variant':<15s} {'MOS':>6s} {'vs off':>7s}")
        print(f"  {'-'*30}")
        baseline = None
        for label, deg_path in variants.items():
            mos = zimtohrli_mos(ref, deg_path)
            if label == "TNS off":
                baseline = mos
            delta = f"{mos - baseline:+.3f}" if baseline is not None else ""
            print(f"  {label:<15s} {mos:6.3f} {delta:>7s}")


def main():
    if "--all" in sys.argv:
        clips = []
        for d in ["test-clips", "open-clips"]:
            full = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    if f.endswith(".wav"):
                        clips.append(os.path.join(full, f))
        for c in clips:
            # Skip mono files (zimtohrli needs matching channels)
            try:
                with wave.open(c) as w:
                    if w.getnchannels() < 2:
                        continue
            except Exception:
                continue
            run_clip(c)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        run_clip(sys.argv[1])
    else:
        print("Usage: python scripts/experiment_tns_vs_tes.py <wav> | --all")


if __name__ == "__main__":
    main()
