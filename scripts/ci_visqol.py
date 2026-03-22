#!/usr/bin/env python3
"""ViSQOL regression test for HQLC.

Usage:
    python scripts/ci_visqol.py                     # test all test-clips/
    python scripts/ci_visqol.py /path/to/track.wav  # score a single track
"""

import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
HQLC_ENC = os.path.join(REPO, "build", "hqlc_enc")
DOCKER_IMAGE = "visqol-local"
DOCKERFILE = os.path.join(SCRIPTS, "Dockerfile.visqol")
BITRATE = 96000
MIN_MOS = 4.5


def run_visqol_mono(ref, deg, tmpdir):
    """Run ViSQOL on a mono pair via docker, return MOS-LQO."""
    # Copy files into tmpdir so docker can see them
    r = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmpdir}:/work",
            DOCKER_IMAGE,
            "--reference_file",
            f"/work/{os.path.basename(ref)}",
            "--degraded_file",
            f"/work/{os.path.basename(deg)}",
            "--similarity_to_quality_model",
            "/app/model/libsvm_nu_svr_model.txt",
        ],
        capture_output=True,
        text=True,
    )
    # Parse MOS-LQO from stdout (more reliable than CSV with docker path mapping)
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"MOS-LQO:\s+([0-9.]+)", line)
        if m:
            return float(m.group(1))
    print(f"  visqol error: {r.stderr[:200]}", file=sys.stderr)
    return 0.0


def to_mono(src, dst, channel):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            src,
            "-af",
            f"pan=mono|c0=c{channel}",
            "-acodec",
            "pcm_s16le",
            dst,
        ],
        check=True,
        capture_output=True,
    )


def visqol_stereo(ref, deg, tmpdir):
    """ViSQOL on stereo: average of L and R."""
    ref_l = os.path.join(tmpdir, "ref_l.wav")
    ref_r = os.path.join(tmpdir, "ref_r.wav")
    deg_l = os.path.join(tmpdir, "deg_l.wav")
    deg_r = os.path.join(tmpdir, "deg_r.wav")
    to_mono(ref, ref_l, 0)
    to_mono(ref, ref_r, 1)
    to_mono(deg, deg_l, 0)
    to_mono(deg, deg_r, 1)
    mos_l = run_visqol_mono(ref_l, deg_l, tmpdir)
    mos_r = run_visqol_mono(ref_r, deg_r, tmpdir)
    return (mos_l + mos_r) / 2.0


def encode_decode(src, tmpdir):
    """Encode through HQLC and return decoded path."""
    name = os.path.splitext(os.path.basename(src))[0]
    decoded = os.path.join(tmpdir, f"{name}_dec.wav")
    subprocess.run(
        [HQLC_ENC, src, decoded, "-b", str(BITRATE)], check=True, capture_output=True
    )
    return decoded


def score_track(src):
    with tempfile.TemporaryDirectory() as tmpdir:
        decoded = encode_decode(src, tmpdir)
        return visqol_stereo(src, decoded, tmpdir)


def main():
    if not os.path.isfile(HQLC_ENC):
        print(f"ERROR: {HQLC_ENC} not found (run cmake --build build)")
        sys.exit(1)

    # Single track mode
    if len(sys.argv) >= 2:
        src = sys.argv[1]
        if not os.path.isfile(src):
            print(f"ERROR: {src} not found")
            sys.exit(1)
        name = os.path.splitext(os.path.basename(src))[0]
        mos = score_track(src)
        print(f"{name}: {mos:.3f}")
        sys.exit(0)

    # Regression test: all test-clips/
    clips_dir = os.path.join(REPO, "test-clips")
    if not os.path.isdir(clips_dir):
        print("ERROR: test-clips/ not found")
        sys.exit(1)

    wavs = sorted(f for f in os.listdir(clips_dir) if f.endswith(".wav"))
    if not wavs:
        print("ERROR: no .wav files in test-clips/")
        sys.exit(1)

    print(f"HQLC ViSQOL regression test ({BITRATE} bps)")
    print("---")

    fails = 0
    scores = []

    for wav in wavs:
        src = os.path.join(clips_dir, wav)
        name = os.path.splitext(wav)[0]
        mos = score_track(src)
        scores.append(mos)
        status = "PASS" if mos >= MIN_MOS else "FAIL"
        if mos < MIN_MOS:
            fails += 1
        print(f"{name:30s} {mos:.3f}  {status}")

    mean = sum(scores) / len(scores)
    print("---")
    print(f"Mean: {mean:.3f} ({len(scores)} tracks)")

    if fails:
        print(f"REGRESSION: {fails} track(s) below {MIN_MOS}")
        sys.exit(1)

    print(f"All tracks above {MIN_MOS} threshold")


if __name__ == "__main__":
    main()
