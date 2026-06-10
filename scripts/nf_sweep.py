#!/usr/bin/env python3
"""Sweep noise-fill experiment knobs and score on burst + perceptual metrics.

Investigates the NF amplitude overshoot: fill amplitude is derived from the
transmitted per-band exponents, which contain the encoder's HF tilt
(~29 dB at the top band at 96k). In quiet HF passages the fill can therefore
land 10-20 dB ABOVE the true signal (steely's P5 bursts). Knobs (made
compile-time overridable in src/quant.c):

    HQLC_NF_DISABLE       fill compiled out — attribution: if steely's P5
                          recovers, NF is confirmed as the burst source
    HQLC_NF_DETILT_PCT    0..100: fraction of the tilt subtracted from the
                          fill envelope (candidate fix; 100 ~= fill follows
                          the true spectral shape)

Scored at 96k on: steely (the target, P5 must rise), castanets (PE — NF
noise also smears ahead of attacks), and Zim on tonal/noise-sensitive guards
(angine, guitar, sopr) since Zimtohrli historically REWARDS HF noise fill —
removing too much may cost Zim even where P5 improves.

Requires the Zim python (3.14): /opt/homebrew/bin/python3.14 scripts/nf_sweep.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr_bench import (load_aligned, snr_db, seg_snr_p5, pre_echo,
                       zim_stereo, _check_zim_python)

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build-nf-sweep"

CONFIGS = [
    ("baseline", ""),
    ("nf-off", "-DHQLC_NF_DISABLE"),
    ("detilt25", "-DHQLC_NF_DETILT_PCT=25"),
    ("detilt50", "-DHQLC_NF_DETILT_PCT=50"),
    ("detilt75", "-DHQLC_NF_DETILT_PCT=75"),
    ("detilt100", "-DHQLC_NF_DETILT_PCT=100"),
]

CLIPS = ["steely48_stereo", "castanets48_stereo", "angine48_stereo",
         "guitar48_stereo", "sopr48_stereo"]
BITRATE = 96000


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


def main():
    _check_zim_python()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        refs = {}
        for stem in CLIPS:
            ref = td / f"ref_{stem}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i",
                 str(REPO / "test-clips2" / f"{stem}.wav"),
                 "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
                 "-t", "30", str(ref)],
                capture_output=True, check=True)
            refs[stem] = ref

        hdr = (f"{'config':<10s} {'steely P5':>9s} {'steely Z':>8s} "
               f"{'cast PE':>8s} {'cast Z':>7s} {'angine Z':>8s} "
               f"{'guitar Z':>8s} {'sopr Z':>7s}")
        print(f"bitrate {BITRATE//1000}k\n\n{hdr}")
        print("─" * len(hdr))
        for name, flags in CONFIGS:
            binary = build_variant(flags)
            res = {}
            for stem, ref in refs.items():
                deg = td / "deg.wav"
                subprocess.run([str(binary), str(ref), str(deg),
                                "-b", str(BITRATE)],
                               capture_output=True, check=True)
                r, d, sr = load_aligned(ref, deg)
                pe, _, _ = pre_echo(r, d, sr)
                res[stem] = {"p5": seg_snr_p5(r, d), "pe": pe,
                             "zim": zim_stereo(ref, deg)}
            st, ca = res["steely48_stereo"], res["castanets48_stereo"]
            print(f"{name:<10s} {st['p5']:>9.2f} {st['zim']:>8.3f} "
                  f"{ca['pe']:>+8.1f} {ca['zim']:>7.3f} "
                  f"{res['angine48_stereo']['zim']:>8.3f} "
                  f"{res['guitar48_stereo']['zim']:>8.3f} "
                  f"{res['sopr48_stereo']['zim']:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
