#!/usr/bin/env python3
"""Diagnose localized codec error bursts (pre-echo, frame artifacts) on one clip.

Roundtrips a clip through build/hqlc with --diag, aligns ref/deg, then answers:
  1. WHEN  — worst segmental windows (timestamps + window SNR), each annotated
             with the encoder state of its frame (TNS order, gain code, gain
             jump, noise factor) so bursts can be tied to encoder decisions.
  2. WHERE — per-octave-band error analysis: for pre-echo regions (before
             detected attacks) and for the worst windows, which frequency
             bands the error energy actually lives in. This is what tells you
             which bins TNS / tilt / window work needs to protect.
  3. WHY   — frame-phase histogram of the worst windows (position mod 512):
             clustering at frame starts implicates frame-boundary
             discontinuities (TNS on/off switching, gain jumps) rather than
             in-frame quantization noise.

Usage:
    python3 scripts/transient_diag.py test-clips2/steely48_stereo.wav -b 96000
    python3 scripts/transient_diag.py test-clips2/castanets48_stereo.wav -b 96000 --top 20
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_bench import (SEG_WIN, PRE_LOOKBACK, PRE_ATTACK_DB, PRE_FLOOR_MS,
                       PRE_REGION_MS, load_aligned, _window_energy)

REPO = Path(__file__).resolve().parent.parent
FRAME = 512

# Octave-ish analysis bands (Hz) for locating error energy in frequency
FREQ_BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
              (4000, 8000), (8000, 16000), (16000, 20000)]


def band_nsr_db(err_seg: np.ndarray, ref_seg: np.ndarray, sr: int) -> list:
    """Error-to-reference ratio (dB) per FREQ_BANDS for one mono segment."""
    n = len(err_seg)
    if n < 64:
        return [None] * len(FREQ_BANDS)
    w = np.hanning(n)
    fe = np.abs(np.fft.rfft(err_seg * w)) ** 2
    fr = np.abs(np.fft.rfft(ref_seg * w)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    out = []
    for lo, hi in FREQ_BANDS:
        m = (freqs >= lo) & (freqs < hi)
        ee, rr = float(fe[m].sum()), float(fr[m].sum())
        out.append(10.0 * np.log10(max(ee, 1e-15) / max(rr, 1e-12)))
    return out


def band_db(seg: np.ndarray, sr: int) -> list:
    """Absolute band energy (dB) per FREQ_BANDS for one mono segment."""
    n = len(seg)
    w = np.hanning(n)
    f = np.abs(np.fft.rfft(seg * w)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    return [10.0 * np.log10(max(float(f[(freqs >= lo) & (freqs < hi)].sum()), 1e-15))
            for lo, hi in FREQ_BANDS]


def load_diag(path: Path) -> list:
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "f" in d:
                frames.append(d)
    return frames


def main():
    ap = argparse.ArgumentParser(description="Per-burst diagnosis of one clip")
    ap.add_argument("clip")
    ap.add_argument("-b", "--bitrate", type=int, default=96000)
    ap.add_argument("--bin", default=str(REPO / "build" / "hqlc"))
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--top", type=int, default=15, help="worst windows to list")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        ref_p, deg_p, diag_p = td / "ref.wav", td / "deg.wav", td / "diag.jsonl"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", args.clip,
             "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
             "-t", str(args.duration), str(ref_p)],
            capture_output=True, check=True)
        subprocess.run(
            [args.bin, str(ref_p), str(deg_p), "-b", str(args.bitrate),
             "--diag", str(diag_p)],
            capture_output=True, check=True)
        ref, deg, sr = load_aligned(ref_p, deg_p)
        diag = load_diag(diag_p)

    err = ref - deg
    mono_ref = ref.mean(axis=1)
    mono_err = err.mean(axis=1)

    # Per-window segmental SNR (same grid as snr_bench P5)
    re = _window_energy(ref)
    ee = _window_energy(err)
    keep = re > SEG_WIN * ref.shape[1] * 1e-6
    snr_w = np.full(len(re), np.inf)
    snr_w[keep] = 10.0 * np.log10(re[keep] / np.maximum(ee[keep], 1e-12))

    tns_on = sum(1 for d in diag if any(o > 0 for o in d["tns"]))
    print(f"clip: {Path(args.clip).stem}  @{args.bitrate//1000}k   "
          f"frames: {len(diag)}  TNS active: {tns_on} ({100*tns_on/max(len(diag),1):.1f}%)")

    # ── 1. Worst windows + encoder state ──────────────────────────────
    order = np.argsort(snr_w)
    print(f"\nWorst {args.top} windows (2.7 ms each):")
    print(f"{'t(s)':>7s} {'winSNR':>7s} {'refdB':>6s}  {'frame':>5s} {'tns':>4s} "
          f"{'gc':>3s} {'dgc':>4s} {'nf':>5s}  dominant err bands")
    for w in order[: args.top]:
        t = w * SEG_WIN / sr
        f_idx = (w * SEG_WIN) // FRAME
        d = diag[f_idx] if f_idx < len(diag) else None
        ref_db = 10.0 * np.log10(max(re[w], 1e-12))
        lo, hi = w * SEG_WIN, (w + 1) * SEG_WIN
        nsr = band_nsr_db(mono_err[lo:hi], mono_ref[lo:hi], sr)
        worst_bands = sorted(range(len(FREQ_BANDS)),
                             key=lambda i: -(nsr[i] if nsr[i] is not None else -99))[:3]
        bands_s = " ".join(f"{FREQ_BANDS[i][0]//1000}-{FREQ_BANDS[i][1]//1000}k"
                           f"({nsr[i]:+.0f})" for i in worst_bands)
        if d:
            dgc = d["gc"] - diag[f_idx - 1]["gc"] if f_idx > 0 else 0
            print(f"{t:7.3f} {snr_w[w]:7.2f} {ref_db:6.1f}  {f_idx:5d} "
                  f"{max(d['tns']):>4d} {d['gc']:>3d} {dgc:+4d} {max(d['nf']):>5d}  {bands_s}")
        else:
            print(f"{t:7.3f} {snr_w[w]:7.2f} {ref_db:6.1f}  {f_idx:5d}     ?  {bands_s}")

    # ── 2. Pre-echo per frequency band (attacks) ──────────────────────
    db = 10.0 * np.log10(np.maximum(re, 1e-12))
    floor_db = 10.0 * np.log10(SEG_WIN * ref.shape[1] * PRE_FLOOR_MS)
    pre_lo = int(sr * PRE_REGION_MS[0] / 1000.0)
    pre_hi = int(sr * PRE_REGION_MS[1] / 1000.0)
    attacks, per_band = [], []
    w = PRE_LOOKBACK
    while w < len(db):
        if db[w] > floor_db and db[w] >= db[w - PRE_LOOKBACK:w].max() + PRE_ATTACK_DB:
            onset = w * SEG_WIN
            if onset - pre_lo >= 0:
                seg_e = mono_err[onset - pre_lo: onset - pre_hi]
                seg_r = mono_ref[onset - pre_lo: onset - pre_hi]
                nsr = band_nsr_db(seg_e, seg_r, sr)
                if nsr[0] is not None:
                    attacks.append(onset / sr)
                    per_band.append(nsr)
                    # TNS state of the frame containing the attack
            w += PRE_LOOKBACK
        else:
            w += 1

    if per_band:
        pb = np.array(per_band)
        print(f"\nPre-echo by frequency band ({len(attacks)} attacks, "
              f"error-vs-ref dB in [-20,-2] ms window; >0 = error louder):")
        print(f"{'band':>10s} {'mean':>7s} {'worst':>7s}")
        for i, (lo, hi) in enumerate(FREQ_BANDS):
            print(f"{lo//1000:>4d}-{hi//1000:<2d}kHz {pb[:, i].mean():>+7.1f} "
                  f"{pb[:, i].max():>+7.1f}")
        # TNS coverage of attack frames
        att_frames = [int(t * sr) // FRAME for t in attacks]
        att_tns = sum(1 for f in att_frames
                      if f < len(diag) and any(o > 0 for o in diag[f]["tns"]))
        # pre-echo lands in the frame BEFORE the attack too — check both
        att_tns_prev = sum(1 for f in att_frames
                           if 0 < f <= len(diag) and any(o > 0 for o in diag[f - 1]["tns"]))
        print(f"\nTNS active on attack frame: {att_tns}/{len(att_frames)}   "
              f"on preceding frame: {att_tns_prev}/{len(att_frames)}")
    else:
        print("\nNo attacks detected — no pre-echo table.")

    # ── 3. Frame-phase histogram of bad windows ───────────────────────
    bad = order[: max(len(order) // 20, args.top)]  # worst 5%
    phases = (np.asarray(bad) * SEG_WIN) % FRAME
    hist = np.bincount(phases // SEG_WIN, minlength=FRAME // SEG_WIN)
    print(f"\nFrame phase of worst 5% windows (uniform ≈ no boundary artifact):")
    for i, c in enumerate(hist):
        print(f"  win {i} [{i*SEG_WIN:>3d}-{(i+1)*SEG_WIN:>3d}): {c:>4d}  "
              + "#" * int(40 * c / max(hist.max(), 1)))


if __name__ == "__main__":
    main()
