#!/usr/bin/env python3
"""Sweep the sub-block transient detector and transient-tilt knobs.

Two phases (run ratio first, then tilt with the chosen ratio):

  ratio   sweep TNS_DETECT_RATIO — reports per-clip firing rates (from the
          --diag "tr" field) plus castanets PE. A good ratio fires ~15-25%%
          on castanets, ~0%% on sustained tonal clips (sopr/ravel), and
          catches steely's smear events around 2.0s and 10.5s.

  tilt    sweep HQLC_TILT_TRANSIENT_PCT at a fixed ratio — transient frames
          get reduced HF pre-emphasis (their spectra are 8-13 dB flatter).
          Scored on PE/P5 + Zim guards (run with python3.14).

Usage:
    python3 scripts/transient_sweep.py ratio
    /opt/homebrew/bin/python3.14 scripts/transient_sweep.py tilt --ratio 8
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_bench import load_aligned, seg_snr_p5, pre_echo

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build-tr-sweep"
BITRATE = 96000

RATIO_VALUES = [4, 8, 16]
TILT_PCT_VALUES = [100, 75, 60, 50]

CLIPS = ["castanets48_stereo", "harpsichord48_stereo", "guitar48_stereo",
         "steely48_stereo", "ravel48_stereo", "sopr48_stereo"]


def build_variant(flags: str) -> Path:
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


def canonicalize(stem: str, td: Path) -> Path:
    ref = td / f"ref_{stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i",
         str(REPO / "test-clips2" / f"{stem}.wav"),
         "-ar", "48000", "-ac", "2", "-sample_fmt", "s16", "-t", "30",
         str(ref)],
        capture_output=True, check=True)
    return ref


def roundtrip(binary: Path, ref: Path, td: Path, diag: bool = False):
    deg = td / "deg.wav"
    diag_p = td / "diag.jsonl"
    cmd = [str(binary), str(ref), str(deg), "-b", str(BITRATE)]
    if diag:
        cmd += ["--diag", str(diag_p)]
    subprocess.run(cmd, capture_output=True, check=True)
    frames = []
    if diag:
        with open(diag_p) as f:
            frames = [json.loads(l) for l in f if '"f"' in l]
    return deg, frames


def fire_rate(frames) -> float:
    if not frames:
        return 0.0
    n = sum(1 for d in frames if any(t for t in d["tr"]))
    return 100.0 * n / len(frames)


def fired_at(frames, t_sec: float) -> bool:
    # frame f covers samples [f*512, (f+1)*512); the MDCT window of frame f
    # also spans the previous frame, so accept f or f+1
    f = int(t_sec * 48000) // 512
    return any(any(t for t in frames[i]["tr"])
               for i in (f, f + 1) if i < len(frames))


def phase_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        refs = {stem: canonicalize(stem, td) for stem in CLIPS}
        hdr = (f"{'ratio':<6s} " +
               " ".join(f"{s.split('48')[0][:9]:>9s}" for s in CLIPS) +
               f" {'cast PE':>8s} {'steely@2.0s':>11s} {'@10.5s':>7s}")
        print(f"firing rate %% per clip at {BITRATE//1000}k\n\n{hdr}")
        print("─" * len(hdr))
        for ratio in RATIO_VALUES:
            binary = build_variant(f"-DTNS_DETECT_RATIO={ratio}")
            rates, cast_pe, st_hits = {}, None, ("?", "?")
            for stem, ref in refs.items():
                deg, frames = roundtrip(binary, ref, td, diag=True)
                rates[stem] = fire_rate(frames)
                if stem.startswith("castanets"):
                    r, d, sr = load_aligned(ref, deg)
                    cast_pe, _, _ = pre_echo(r, d, sr)
                if stem.startswith("steely"):
                    st_hits = ("y" if fired_at(frames, 2.05) else "n",
                               "y" if fired_at(frames, 10.52) else "n")
            print(f"{ratio:<6d} " +
                  " ".join(f"{rates[s]:>9.1f}" for s in CLIPS) +
                  f" {cast_pe:>+8.1f} {st_hits[0]:>11s} {st_hits[1]:>7s}",
                  flush=True)


def phase_tilt(ratio: int):
    from snr_bench import zim_stereo, _check_zim_python
    _check_zim_python()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        refs = {stem: canonicalize(stem, td) for stem in CLIPS}
        hdr = (f"{'tilt%':<6s} {'cast PE':>8s} {'cast P5':>8s} {'cast Z':>7s} "
               f"{'harp PE':>8s} {'steely P5':>9s} {'steely Z':>8s} "
               f"{'guitar Z':>8s} {'sopr Z':>7s} {'ravel Z':>8s}")
        print(f"ratio={ratio}, {BITRATE//1000}k\n\n{hdr}")
        print("─" * len(hdr))
        for pct in TILT_PCT_VALUES:
            binary = build_variant(
                f"-DTNS_DETECT_RATIO={ratio} -DHQLC_TILT_TRANSIENT_PCT={pct}")
            res = {}
            for stem, ref in refs.items():
                deg, _ = roundtrip(binary, ref, td)
                r, d, sr = load_aligned(ref, deg)
                pe, _, _ = pre_echo(r, d, sr)
                res[stem] = {"pe": pe, "p5": seg_snr_p5(r, d),
                             "zim": zim_stereo(ref, deg)}
            c = res["castanets48_stereo"]
            h = res["harpsichord48_stereo"]
            st = res["steely48_stereo"]
            print(f"{pct:<6d} {c['pe']:>+8.1f} {c['p5']:>8.2f} "
                  f"{c['zim']:>7.3f} {h['pe']:>+8.1f} {st['p5']:>9.2f} "
                  f"{st['zim']:>8.3f} {res['guitar48_stereo']['zim']:>8.3f} "
                  f"{res['sopr48_stereo']['zim']:>7.3f} "
                  f"{res['ravel48_stereo']['zim']:>8.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="transient detector/tilt sweep")
    ap.add_argument("phase", choices=["ratio", "tilt"])
    ap.add_argument("--ratio", type=int, default=8,
                    help="TNS_DETECT_RATIO for the tilt phase")
    args = ap.parse_args()
    if args.phase == "ratio":
        phase_ratio()
    else:
        phase_tilt(args.ratio)


if __name__ == "__main__":
    main()
