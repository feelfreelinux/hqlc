#!/usr/bin/env python3
"""ViSQOL benchmark for HQLC and reference codecs.

Usage:
    python scripts/visqol_bench.py single track.wav
    python scripts/visqol_bench.py dir --input /path/to/wavs
    python scripts/visqol_bench.py sqam -b 128000 -c hqlc,opus,aac
    python scripts/visqol_bench.py musdb -b 96000 --split test
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
HQLC_ENC = REPO / "build" / "hqlc_enc"
DOCKER_IMAGE = "visqol-local"
DOCKERFILE = SCRIPTS / "Dockerfile.visqol"
MAX_DURATION = 30

LC3_ENC = None
LC3_DEC = None


# ── Audio helpers ──

def to_48k_wav(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
         "-t", str(MAX_DURATION), str(dst)],
        capture_output=True, check=True)


def to_mono(src, dst, channel=0):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"pan=mono|c0=c{channel}", "-acodec", "pcm_s16le", str(dst)],
        capture_output=True, check=True)


def extract_musdb_mix(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-map", "0:a:0", "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
         "-t", str(MAX_DURATION), str(dst)],
        capture_output=True, check=True)


# ── ViSQOL via docker ──

def visqol_mono(ref, deg, tmpdir):
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{tmpdir}:/work", DOCKER_IMAGE,
         "--reference_file", f"/work/{os.path.basename(ref)}",
         "--degraded_file", f"/work/{os.path.basename(deg)}",
         "--similarity_to_quality_model", "/app/model/libsvm_nu_svr_model.txt"],
        capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"MOS-LQO:\s+([0-9.]+)", line)
        if m:
            return float(m.group(1))
    print(f"  visqol error: {r.stderr[:200]}", file=sys.stderr)
    return 0.0


def visqol_stereo(ref, deg, tmpdir):
    td = Path(tmpdir)
    to_mono(ref, td / "ref_l.wav", 0)
    to_mono(ref, td / "ref_r.wav", 1)
    to_mono(deg, td / "deg_l.wav", 0)
    to_mono(deg, td / "deg_r.wav", 1)
    mos_l = visqol_mono(td / "ref_l.wav", td / "deg_l.wav", tmpdir)
    mos_r = visqol_mono(td / "ref_r.wav", td / "deg_r.wav", tmpdir)
    return (mos_l + mos_r) / 2.0


# ── Codec encode/decode ──

def encode_hqlc(ref, deg, bitrate, tmp):
    subprocess.run([str(HQLC_ENC), str(ref), str(deg), "-b", str(bitrate)],
                   capture_output=True, check=True)


def _ffmpeg_encode(ref, deg, bitrate, codec_args, ext):
    intermediate = str(deg).replace(".wav", ext)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(ref)]
                   + codec_args + [intermediate],
                   capture_output=True, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", intermediate,
                    "-acodec", "pcm_s16le", str(deg)],
                   capture_output=True, check=True)


def encode_opus(ref, deg, bitrate, tmp):
    _ffmpeg_encode(ref, deg, bitrate,
                   ["-c:a", "libopus", "-b:a", str(bitrate), "-vbr", "off"], ".ogg")


def encode_aac(ref, deg, bitrate, tmp):
    _ffmpeg_encode(ref, deg, bitrate, ["-c:a", "aac", "-b:a", str(bitrate)], ".m4a")


def encode_mp3(ref, deg, bitrate, tmp):
    _ffmpeg_encode(ref, deg, bitrate,
                   ["-c:a", "libmp3lame", "-b:a", str(bitrate)], ".mp3")


def encode_lc3(ref, deg, bitrate, tmp):
    mono_bitrate = bitrate // 2
    ref_l, ref_r = tmp / "lc3_ref_l.wav", tmp / "lc3_ref_r.wav"
    to_mono(ref, ref_l, 0)
    to_mono(ref, ref_r, 1)

    lc3_lib = str(Path(LC3_ENC).parent)
    env = dict(os.environ, DYLD_LIBRARY_PATH=lc3_lib, LD_LIBRARY_PATH=lc3_lib)

    for mono, lc3f in [(ref_l, tmp / "l.lc3"), (ref_r, tmp / "r.lc3")]:
        subprocess.run([str(LC3_ENC), "-b", str(mono_bitrate), str(mono), str(lc3f)],
                       capture_output=True, check=True, env=env)

    dec_l, dec_r = tmp / "lc3_dec_l.wav", tmp / "lc3_dec_r.wav"
    for lc3f, dec in [(tmp / "l.lc3", dec_l), (tmp / "r.lc3", dec_r)]:
        subprocess.run([str(LC3_DEC), str(lc3f), str(dec)],
                       capture_output=True, check=True, env=env)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(dec_l), "-i", str(dec_r),
         "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
         "-map", "[a]", "-acodec", "pcm_s16le", str(deg)],
        capture_output=True, check=True)


CODECS = {"hqlc": encode_hqlc, "opus": encode_opus, "aac": encode_aac,
          "mp3": encode_mp3, "lc3": encode_lc3}


# ── Track processing ──
# prep_mode: "wav" = to_48k_wav, "musdb" = extract_musdb_mix

def process_track(name, src_path, prep_mode, codec, bitrate):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_wav = tmp / "ref.wav"

        try:
            if prep_mode == "musdb":
                extract_musdb_mix(src_path, ref_wav)
            else:
                to_48k_wav(src_path, ref_wav)
        except Exception as e:
            return name, codec, 0.0, f"prep: {str(e)[:60]}"

        deg_wav = tmp / "deg.wav"
        try:
            CODECS[codec](ref_wav, deg_wav, bitrate, tmp)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e)
            return name, codec, 0.0, f"encode: {err[:60]}"

        if not deg_wav.exists():
            return name, codec, 0.0, "no output"

        try:
            mos = visqol_stereo(ref_wav, deg_wav, tmpdir)
        except Exception as e:
            return name, codec, 0.0, f"visqol: {str(e)[:60]}"

        return name, codec, mos, "ok"


# ── Dataset loaders (each returns [(name, src_path, prep_mode), ...]) ──

def load_sqam():
    sqam_dir = REPO / "datasets" / "SQAM_FLAC_00s9l4"
    if not sqam_dir.exists():
        sqam_dir = REPO / "sqam"
    m3u = sqam_dir / "EBU SQAM.m3u"
    if not m3u.exists():
        sys.exit(f"Error: {m3u} not found")

    tracks = []
    for line in m3u.read_text(errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = sqam_dir / line
            if path.exists():
                tracks.append((Path(line).stem, str(path), "wav"))
    return tracks


def load_musdb(split):
    musdb_dir = REPO / "datasets" / "musdb18"
    if not musdb_dir.exists():
        musdb_dir = REPO / "musdb18"
    mp4s = []
    if split in ("test", "both"):
        mp4s += sorted((musdb_dir / "test").glob("*.stem.mp4"))
    if split in ("train", "both"):
        mp4s += sorted((musdb_dir / "train").glob("*.stem.mp4"))
    if not mp4s:
        sys.exit("Error: no MUSDB18 tracks found")
    return [(p.stem.replace(".stem", ""), str(p), "musdb") for p in mp4s]


def load_dir(input_dir):
    wavs = sorted(Path(input_dir).glob("*.wav"))
    if not wavs:
        sys.exit(f"Error: no .wav files in {input_dir}")
    return [(w.stem, str(w), "wav") for w in wavs]


def _init_pool_worker(lc3_enc, lc3_dec):
    global LC3_ENC, LC3_DEC
    LC3_ENC, LC3_DEC = lc3_enc, lc3_dec


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="ViSQOL benchmark for HQLC")
    sub = parser.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("-b", "--bitrate", type=int, default=96000)
        p.add_argument("-j", "--jobs", type=int, default=4)
        p.add_argument("-c", "--codecs", type=str, default="hqlc")
        p.add_argument("-o", "--output", type=str, default=None)
        p.add_argument("--lc3-enc", type=str, default=None)
        p.add_argument("--lc3-dec", type=str, default=None)

    p = sub.add_parser("sqam")
    common(p)
    p = sub.add_parser("musdb")
    p.add_argument("--split", default="test", choices=["test", "train", "both"])
    common(p)
    p = sub.add_parser("dir")
    p.add_argument("--input", required=True)
    common(p)
    p = sub.add_parser("single")
    p.add_argument("wav")
    common(p)

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    global LC3_ENC, LC3_DEC
    LC3_ENC = args.lc3_enc
    LC3_DEC = args.lc3_dec

    codec_list = [c.strip() for c in args.codecs.split(",")]
    for c in codec_list:
        if c not in CODECS:
            sys.exit(f"Unknown codec: {c}. Available: {','.join(CODECS)}")
    if "hqlc" in codec_list and not HQLC_ENC.exists():
        sys.exit(f"Error: {HQLC_ENC} not found")
    if "lc3" in codec_list and (not LC3_ENC or not Path(LC3_ENC).exists()):
        sys.exit("Error: --lc3-enc path required for lc3")
    if "lc3" in codec_list and (not LC3_DEC or not Path(LC3_DEC).exists()):
        sys.exit("Error: --lc3-dec path required for lc3")

    # Single track
    if args.cmd == "single":
        if not os.path.isfile(args.wav):
            sys.exit(f"Error: {args.wav} not found")
        for codec in codec_list:
            _, _, mos, status = process_track(
                Path(args.wav).stem, args.wav, "wav", codec, args.bitrate)
            print(f"{Path(args.wav).stem}: {mos:.3f} ({codec})" if status == "ok"
                  else f"{Path(args.wav).stem}: {status} ({codec})")
        return

    # Load dataset
    if args.cmd == "sqam":
        tracks = load_sqam()
    elif args.cmd == "musdb":
        tracks = load_musdb(args.split)
    elif args.cmd == "dir":
        tracks = load_dir(args.input)

    print(f"{len(tracks)} tracks, {args.bitrate} bps, codecs: {', '.join(codec_list)}\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs,
                             initializer=_init_pool_worker,
                             initargs=(LC3_ENC, LC3_DEC)) as pool:
        futures = {}
        for name, src, prep_mode in tracks:
            for codec in codec_list:
                f = pool.submit(process_track, name, src, prep_mode, codec, args.bitrate)
                futures[f] = (name, codec)

        for future in as_completed(futures):
            name, codec, mos, status = future.result()
            if status == "ok":
                print(f"  {codec:>5s}  {mos:.3f}  {name}", flush=True)
            else:
                print(f"  {codec:>5s}  {status}  {name}", flush=True)
            results.append((name, codec, mos, status))

    # Summary
    print(f"\n{'Codec':>8s}  {'Mean':>6s}  {'Min':>6s}  {'Max':>6s}  {'N':>3s}")
    print("─" * 40)
    for codec in codec_list:
        scores = [m for _, c, m, s in results if c == codec and s == "ok" and m > 0]
        if scores:
            print(f"{codec:>8s}  {sum(scores)/len(scores):6.3f}  {min(scores):6.3f}  "
                  f"{max(scores):6.3f}  {len(scores):3d}")

    for codec in codec_list:
        ok = sorted([(n, m) for n, c, m, s in results if c == codec and s == "ok" and m > 0],
                    key=lambda x: x[1])
        if ok:
            print(f"\n  {codec} bottom 3:")
            for n, m in ok[:3]:
                print(f"    {m:.3f}  {n}")

    # CSV
    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track", "bitrate", "codec", "mos_lqo", "status"])
            for name, codec, mos, status in sorted(results):
                w.writerow([name, args.bitrate, codec, f"{mos:.4f}", status])
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
