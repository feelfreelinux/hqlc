#!/usr/bin/env python3
"""Sweep TNS compile-time settings and score them on the transient metrics.

Rebuilds the codec into build-tns-sweep/ with -D overrides for
TNS_START_BIN / TNS_MAX_ORDER / TNS_ATTACK_RATIO (made #ifndef-overridable
for this purpose), then scores each variant with the snr_bench metrics on a
transient-focused clip set at 96k:

    castanets    the pre-echo benchmark (PE is THE number to move)
    harpsichord  borderline pre-echo (PE +0.7 baseline)
    guitar       6 attacks, healthy baseline (must not regress)
    steely       HF burst clip (P5 guard — bursts are NF/tilt, not TNS,
                 but always-on TNS must not make them worse)
    ravel, sopr  tonal guards (no attacks; SNR/P5 must hold — always-on
                 TNS risks distorting sustained content)

Bitstream compatibility: encoder+decoder live in the same binary, so
START_BIN/order changes roundtrip fine within a variant. A winning config
still needs the full bench (incl. --visqol --zim) before adoption.

Usage:
    python3 scripts/tns_sweep.py            # all configs
    python3 scripts/tns_sweep.py --bitrate 128000
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_bench import load_aligned, snr_db, seg_snr_p5, pre_echo

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build-tns-sweep"

CONFIGS = [
    # (name, {macro: value})
    ("baseline", {}),
    ("start20", {"TNS_START_BIN": 20}),            # ~940 Hz
    ("start9", {"TNS_START_BIN": 9}),              # ~420 Hz
    ("order8", {"TNS_MAX_ORDER": 8}),
    ("always", {"TNS_ATTACK_RATIO": 0}),           # gain-criterion gated only
    ("o8+s20", {"TNS_MAX_ORDER": 8, "TNS_START_BIN": 20}),
    ("o8+s20+alw", {"TNS_MAX_ORDER": 8, "TNS_START_BIN": 20,
                    "TNS_ATTACK_RATIO": 0}),
]

CLIPS = ["castanets48_stereo", "harpsichord48_stereo", "guitar48_stereo",
         "steely48_stereo", "ravel48_stereo", "sopr48_stereo"]


def build_variant(defines: dict) -> Path:
    flags = " ".join(f"-D{k}={v}" for k, v in defines.items())
    subprocess.run(
        ["cmake", "-B", str(BUILD), "-S", str(REPO),
         "-DCMAKE_BUILD_TYPE=Release", "-DHQLC_TOOLS=ON", "-DHQLC_TESTS=OFF",
         f"-DCMAKE_C_FLAGS={flags}"],
        capture_output=True, check=True)
    subprocess.run(
        ["cmake", "--build", str(BUILD), "--target", "hqlc_cli", "-j",
         "--clean-first"],
        capture_output=True, check=True)
    return BUILD / "hqlc"


def score_clip(binary: Path, ref: Path, bitrate: int, tmp: Path) -> dict:
    deg = tmp / f"deg_{ref.stem}.wav"
    subprocess.run([str(binary), str(ref), str(deg), "-b", str(bitrate)],
                   capture_output=True, check=True)
    r, d, sr = load_aligned(ref, deg)
    pe_w, pe_m, n_att = pre_echo(r, d, sr)
    return {"snr": snr_db(r, d), "p5": seg_snr_p5(r, d),
            "pe": pe_w, "pe_mean": pe_m, "n": n_att}


def main():
    ap = argparse.ArgumentParser(description="TNS settings sweep")
    ap.add_argument("--bitrate", type=int, default=96000)
    ap.add_argument("--clips-dir", default=str(REPO / "test-clips2"))
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # Canonicalize clips once, reuse across all variants
        refs = []
        for stem in CLIPS:
            src = Path(args.clips_dir) / f"{stem}.wav"
            ref = td / f"ref_{stem}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                 "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
                 "-t", str(args.duration), str(ref)],
                capture_output=True, check=True)
            refs.append(ref)

        print(f"bitrate {args.bitrate//1000}k — PE(worst)/P5 per clip, "
              f"SNR for tonal guards\n")
        short = {"castanets48_stereo": "castanets", "harpsichord48_stereo": "harps",
                 "guitar48_stereo": "guitar", "steely48_stereo": "steely",
                 "ravel48_stereo": "ravel", "sopr48_stereo": "sopr"}
        hdr = (f"{'config':<12s} {'cast PE':>8s} {'cast P5':>8s} "
               f"{'harp PE':>8s} {'guit PE':>8s} {'steely P5':>9s} "
               f"{'ravel SNR':>9s} {'sopr SNR':>8s}")
        print(hdr)
        print("─" * len(hdr))
        for name, defines in CONFIGS:
            binary = build_variant(defines)
            res = {}
            for ref in refs:
                stem = ref.stem[4:]  # strip "ref_"
                res[short[stem]] = score_clip(binary, ref, args.bitrate, td)
            c, h, g = res["castanets"], res["harps"], res["guitar"]
            fmt_pe = lambda r: f"{r['pe']:+8.1f}" if r["pe"] is not None else "       —"
            print(f"{name:<12s} {fmt_pe(c)} {c['p5']:>8.2f} {fmt_pe(h)} "
                  f"{fmt_pe(g)} {res['steely']['p5']:>9.2f} "
                  f"{res['ravel']['snr']:>9.2f} {res['sopr']['snr']:>8.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
