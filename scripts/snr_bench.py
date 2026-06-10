#!/usr/bin/env python3
"""Quality benchmark of the HQLC C build across bitrates.

This is THE benchmark to run when evaluating a codec change. It roundtrips
each clip through the existing build/hqlc binary at several bitrates and
reports waveform + transient-sensitive metrics, optionally plus perceptual
ViSQOL / Zimtohrli MOS. It does NOT rebuild — point --bin at whatever binary
you want to score, so the same harness compares different solutions fairly.

WHAT IT MEASURES
    SNR (dB)  time-domain waveform SNR after cross-correlation alignment
              (the codec adds a ~512-sample MDCT delay). Cheap, no deps
              beyond ffmpeg+numpy. Good fast proxy + sanity check; note it
              under-reads on transients (castanets ~6-9 dB stereo is normal,
              not a bug — transients wreck waveform SNR).
    P5 (dB)   5th percentile of segmental SNR (128-sample / 2.7 ms windows,
              silence-gated, clamped to [-20, 60] dB). Exposes localized
              error bursts (pre-echo, TNS on/off boundaries, NF overshoot)
              that the full-clip mean averages away.
    PE (dB)   pre-echo: error-to-reference energy ratio in the 20..2 ms
              window BEFORE each attack detected in the reference. Acoustic
              pre-masking only covers ~2 ms, so codec noise in that window is
              audible by definition. Reported as worst over attacks (and mean
              + attack count in the CSV); > 0 dB means the codec error is
              LOUDER than the actual signal content just before an attack.
              "—" if the clip has no detected attacks.
    ViSQOL    MOS-LQO via the visqol-local docker image, per channel, averaged.
    Zim       Zimtohrli MOS via mos/quality_metrics.py.
    Both perceptual metrics use >=20 ms analysis windows and mean-aggregate,
    so they are nearly blind to pre-echo and other localized artifacts —
    that is exactly what P5/PE are for. Judge transient work on P5/PE,
    steady-state work on ViSQOL/Zim, and keep SNR as a sanity check.

TYPICAL WORKFLOW (comparing solutions)
    1. cmake --build build --target hqlc_cli -j           # build solution A
    2. python scripts/snr_bench.py --visqol --zim -o a.csv
    3. ...make a change, rebuild...                        # solution B
    4. python scripts/snr_bench.py --visqol --zim -o b.csv
    5. diff the summary tables / CSVs. Meaningful deltas: > ~0.05 ViSQOL,
       > ~0.5 dB SNR, > ~1 dB P5, > ~2 dB PE on a transient clip; smaller
       is noise. PE/P5 are per-clip metrics — look at the worst clips, not
       only the means.
    (Or keep two binaries and run twice with --bin build/hqlc vs --bin other.)

REFERENCE BASELINES (test-clips2, 9 clips, feature/normalize-spectral-coeffs,
2026-06-10, full run):
    96k :  SNR 16.54  ViSQOL 4.544  Zim 4.680
    128k:  SNR 20.34  ViSQOL 4.624  Zim 4.833
    196k:  SNR 25.57  ViSQOL 4.682  Zim 4.941

BACKEND SETUP
    SNR/P5/PE: any Python 3.9+ with numpy/scipy + ffmpeg. e.g. the macOS
               system python3 is fine.
    Zimtohrli (--zim):  the prebuilt mos/zimtohrli/build/_pyohrli.so is a
               CPython extension linked against ONE interpreter (currently
               Python 3.14). Run with THAT python or it segfaults at call time:
                   /opt/homebrew/bin/python3.14 scripts/snr_bench.py --zim ...
               The script checks this and tells you the right interpreter.
    ViSQOL   (--visqol): needs the docker image once:
                   docker build -f scripts/Dockerfile.visqol -t visqol-local .
               (without it, ViSQOL reports 0.000)

Usage:
    python3 scripts/snr_bench.py                  # SNR/P5/PE @ 96k/128k/196k
    python3 scripts/snr_bench.py --bitrates 96000,128000
    python3.14 scripts/snr_bench.py --visqol --zim   # add perceptual metrics
    python3 scripts/snr_bench.py --bin build/hqlc --clips-dir test-clips2 -o snr.csv
"""

import argparse
import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

REPO = Path(__file__).resolve().parent.parent
DOCKER_IMAGE = "visqol-local"

# Segmental window: 128 samples = 2.67 ms at 48 kHz. Shared by P5 and the
# pre-echo attack detector so both see the same time grid.
SEG_WIN = 128
# Per-window SNR clamp (standard segSNR practice — keeps silence/perfect
# windows from dominating the distribution).
SEG_SNR_CLAMP = (-20.0, 60.0)
# Reference windows quieter than this mean-square level are skipped by P5:
# SNR inside near-silence is meaningless and perceptually irrelevant.
SEG_SILENCE_MS = 1e-6  # -60 dBFS
# Attack detection: window must be this far above the loudest of the
# preceding PRE_LOOKBACK windows, and above an absolute floor.
PRE_ATTACK_DB = 12.0
PRE_LOOKBACK = 8  # windows (~21 ms)
PRE_FLOOR_MS = 1e-4  # -40 dBFS: don't call noise-floor wiggle an attack
# Pre-echo measurement region before the attack onset. Starts at -20 ms
# (one MDCT half-frame is 10.7 ms, so smear never reaches further) and stops
# at -2 ms (acoustic pre-masking hides anything closer to the attack).
PRE_REGION_MS = (20.0, 2.0)
PRE_RATIO_CLAMP = 40.0  # dB cap when the reference is dead silent pre-attack


# ── Signal loading / alignment ─────────────────────────────────────────

def _read_norm(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(str(path))
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return sr, x


def load_aligned(ref_path: Path, deg_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Load both files and align them on the codec's MDCT delay.

    Returns (ref, deg, sample_rate) trimmed to equal length, channels in
    columns. Lag is found by peak cross-correlation on channel 0, searched
    coarse (step 16) over [0, 1024] then refined ±16.
    """
    sr_r, ref = _read_norm(ref_path)
    _, deg = _read_norm(deg_path)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    a, b = ref[:, 0], deg[:, 0]
    best_lag, best_corr = 0, -np.inf
    for lag in range(0, 1025, 16):
        if lag >= n:
            break
        c = float(np.dot(a[: n - lag], b[lag:]))
        if c > best_corr:
            best_corr, best_lag = c, lag
    for lag in range(max(0, best_lag - 16), min(n - 1, best_lag + 16)):
        c = float(np.dot(a[: n - lag], b[lag:]))
        if c > best_corr:
            best_corr, best_lag = c, lag
    return ref[: n - best_lag], deg[best_lag:], sr_r


def _window_energy(x: np.ndarray) -> np.ndarray:
    """Sum-of-squares per SEG_WIN window, summed across channels."""
    n = (len(x) // SEG_WIN) * SEG_WIN
    if n == 0:
        return np.empty(0)
    return (x[:n] ** 2).sum(axis=1).reshape(-1, SEG_WIN).sum(axis=1)


# ── Metrics (on aligned arrays) ────────────────────────────────────────

def snr_db(ref: np.ndarray, deg: np.ndarray) -> float:
    err = ref - deg
    s = float(np.sum(ref * ref))
    e = float(np.sum(err * err))
    if e <= 0 or s <= 0:
        return 0.0
    return 10.0 * np.log10(s / e)


def seg_snr_p5(ref: np.ndarray, deg: np.ndarray) -> float:
    """5th percentile of per-window SNR — the 'worst moments' metric."""
    re = _window_energy(ref)
    ee = _window_energy(ref - deg)
    if len(re) == 0:
        return 0.0
    keep = re > SEG_WIN * ref.shape[1] * SEG_SILENCE_MS
    if not keep.any():
        return 0.0
    snr = 10.0 * np.log10(re[keep] / np.maximum(ee[keep], 1e-12))
    return float(np.percentile(np.clip(snr, *SEG_SNR_CLAMP), 5))


def pre_echo(ref: np.ndarray, deg: np.ndarray, sr: int):
    """Pre-echo around reference attacks.

    Detects attacks (window jumping PRE_ATTACK_DB above the preceding
    ~21 ms, above an absolute floor), then measures codec error energy vs
    reference energy in the [-20 ms, -2 ms] window before each onset.

    Returns (worst_db, mean_db, n_attacks); (None, None, 0) if no attacks.
    """
    err = ref - deg
    e_ref = _window_energy(ref)
    if len(e_ref) <= PRE_LOOKBACK:
        return None, None, 0
    db = 10.0 * np.log10(np.maximum(e_ref, 1e-12))
    floor_db = 10.0 * np.log10(SEG_WIN * ref.shape[1] * PRE_FLOOR_MS)
    pre_lo = int(sr * PRE_REGION_MS[0] / 1000.0)
    pre_hi = int(sr * PRE_REGION_MS[1] / 1000.0)

    ratios = []
    w = PRE_LOOKBACK
    while w < len(db):
        if db[w] > floor_db and db[w] >= db[w - PRE_LOOKBACK:w].max() + PRE_ATTACK_DB:
            onset = w * SEG_WIN
            lo, hi = onset - pre_lo, onset - pre_hi
            if lo >= 0:
                e_e = float((err[lo:hi] ** 2).sum())
                # Floor the reference energy at -100 dBFS so silence before an
                # attack (where pre-echo is MOST audible) caps at +40 dB
                # instead of dividing by zero.
                e_r = max(float((ref[lo:hi] ** 2).sum()),
                          (hi - lo) * ref.shape[1] * 1e-10)
                r = 10.0 * np.log10(max(e_e, 1e-12) / e_r)
                ratios.append(min(r, PRE_RATIO_CLAMP))
            w += PRE_LOOKBACK  # refractory: don't re-trigger on the decay
        else:
            w += 1
    if not ratios:
        return None, None, 0
    return float(max(ratios)), float(np.mean(ratios)), len(ratios)


# ── Perceptual backends ────────────────────────────────────────────────

def to_mono(src: Path, dst: Path, channel: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", f"pan=mono|c0=c{channel}", "-acodec", "pcm_s16le", str(dst)],
        capture_output=True, check=True,
    )


def visqol_mono(ref: Path, deg: Path, tmpdir: str) -> float:
    import re
    r = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{tmpdir}:/work", DOCKER_IMAGE,
         "--reference_file", f"/work/{ref.name}",
         "--degraded_file", f"/work/{deg.name}",
         "--similarity_to_quality_model", "/app/model/libsvm_nu_svr_model.txt"],
        capture_output=True, text=True,
    )
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"MOS-LQO:\s+([0-9.]+)", line)
        if m:
            return float(m.group(1))
    return 0.0


def visqol_stereo(ref: Path, deg: Path, tmpdir: str) -> float:
    td = Path(tmpdir)
    to_mono(ref, td / "ref_l.wav", 0)
    to_mono(ref, td / "ref_r.wav", 1)
    to_mono(deg, td / "deg_l.wav", 0)
    to_mono(deg, td / "deg_r.wav", 1)
    a = visqol_mono(td / "ref_l.wav", td / "deg_l.wav", tmpdir)
    b = visqol_mono(td / "ref_r.wav", td / "deg_r.wav", tmpdir)
    return (a + b) / 2.0


_ZIM = None


def _zim_metric():
    """Lazy-init Zimtohrli metric (per-process)."""
    global _ZIM
    if _ZIM is not None:
        return _ZIM
    sys.path.insert(0, str(REPO / "mos"))
    from quality_metrics import QualityConfig, get_quality_metric
    _ZIM = get_quality_metric(QualityConfig(backend="zimtohrli"))
    return _ZIM


def zim_stereo(ref_path: Path, deg_path: Path) -> float:
    sr_r, ref = wavfile.read(str(ref_path))
    sr_d, deg = wavfile.read(str(deg_path))
    if sr_r != 48000 or sr_d != 48000:
        return 0.0
    if ref.dtype.kind == "i":
        ref = ref.astype(np.float32) / np.iinfo(ref.dtype).max
    if deg.dtype.kind == "i":
        deg = deg.astype(np.float32) / np.iinfo(deg.dtype).max
    if ref.ndim == 1:
        ref = np.stack([ref, ref], axis=1)
    if deg.ndim == 1:
        deg = np.stack([deg, deg], axis=1)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    metric = _zim_metric()
    return float(metric.mos_stereo_lr_avg(
        np.ascontiguousarray(ref[:, 0]),
        np.ascontiguousarray(ref[:, 1]),
        np.ascontiguousarray(deg[:, 0]),
        np.ascontiguousarray(deg[:, 1]),
    ))


# ── Per-clip task ─────────────────────────────────────────────────────

def process_clip(args: tuple) -> dict:
    bitrate, clip_name, clip_path, hqlc_bin, duration, do_visqol, do_zim = args
    row = {"bitrate": bitrate, "clip": clip_name,
           "snr": 0.0, "seg_p5": 0.0,
           "pe_worst": None, "pe_mean": None, "n_attacks": 0,
           "visqol": 0.0, "zim": 0.0, "status": "ok"}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ref = tmpdir / "ref.wav"
        deg = tmpdir / "deg.wav"
        # Canonicalize to 48k s16 stereo so all metrics are comparable to baselines.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip_path),
             "-ar", "48000", "-ac", "2", "-sample_fmt", "s16",
             "-t", str(duration), str(ref)],
            capture_output=True, check=True,
        )
        try:
            subprocess.run([str(hqlc_bin), str(ref), str(deg), "-b", str(bitrate)],
                           capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            row["status"] = f"encode: {e.stderr.decode(errors='replace')[:80]}"
            return row
        if not deg.exists():
            row["status"] = "no output"
            return row
        try:
            r, d, sr = load_aligned(ref, deg)
            row["snr"] = snr_db(r, d)
            row["seg_p5"] = seg_snr_p5(r, d)
            row["pe_worst"], row["pe_mean"], row["n_attacks"] = pre_echo(r, d, sr)
        except Exception as e:
            row["status"] = f"metrics: {str(e)[:60]}"
            return row
        if do_visqol:
            try:
                row["visqol"] = visqol_stereo(ref, deg, str(tmpdir))
            except Exception as e:
                print(f"  visqol error on {clip_name}: {e}", file=sys.stderr)
        if do_zim:
            try:
                row["zim"] = zim_stereo(ref, deg)
            except Exception as e:
                print(f"  zim error on {clip_name}: {e}", file=sys.stderr)
    return row


# ── Zimtohrli interpreter guard ────────────────────────────────────────
# The prebuilt mos/zimtohrli/build/_pyohrli.so is a CPython extension linked
# against one specific Python (currently 3.14). Loading it under a different
# interpreter import-succeeds but SEGFAULTS at first call, which surfaces only
# as an opaque BrokenProcessPool. Detect the mismatch up front with a clear
# message instead.

def _check_zim_python() -> None:
    so = REPO / "mos" / "zimtohrli" / "build" / "_pyohrli.so"
    if not so.exists():
        sys.exit(f"--zim: {so} not found — build pyohrli (cmake in mos/zimtohrli) first.")
    want = None
    try:  # best-effort: read the linked Python framework version via otool (macOS)
        import re
        out = subprocess.run(["otool", "-L", str(so)], capture_output=True, text=True).stdout
        m = re.search(r"Python\.framework/Versions/(\d+\.\d+)/Python", out)
        if m:
            want = m.group(1)
    except Exception:
        return  # can't determine (e.g. non-macOS) — let it run and fail naturally
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if want and want != have:
        sys.exit(
            f"--zim needs Python {want} ({so.name} is linked against it), "
            f"but this is Python {have}.\n"
            f"Re-run with that interpreter, e.g.:\n"
            f"    /opt/homebrew/bin/python{want} {' '.join(sys.argv)}")


# ── Driver ─────────────────────────────────────────────────────────────

def _fmt_pe(worst, n) -> str:
    if worst is None:
        return "    —    "
    return f"{worst:+6.1f}/{n:<3d}"


def main():
    ap = argparse.ArgumentParser(
        description="SNR / segmental-P5 / pre-echo (+ optional perceptual) bench of build/hqlc")
    ap.add_argument("--bin", default=str(REPO / "build" / "hqlc"),
                    help="path to the hqlc binary (default: build/hqlc)")
    ap.add_argument("--clips-dir", default=str(REPO / "test-clips2"))
    ap.add_argument("--bitrates", default="96000,128000,196000")
    ap.add_argument("--duration", type=int, default=30, help="seconds per clip (default 30)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--visqol", action="store_true", help="also run ViSQOL (needs docker image)")
    ap.add_argument("--zim", action="store_true", help="also run Zimtohrli (needs mos/ backend)")
    ap.add_argument("-o", "--output", default=None, help="write per-clip CSV")
    args = ap.parse_args()

    hqlc_bin = Path(args.bin)
    if not hqlc_bin.exists():
        sys.exit(f"Error: {hqlc_bin} not found — build it first (cmake --build build --target hqlc_cli)")
    clips = sorted(Path(args.clips_dir).glob("*.wav"))
    if not clips:
        sys.exit(f"No .wav clips in {args.clips_dir}")
    if args.zim:
        _check_zim_python()
    bitrates = [int(b) for b in args.bitrates.split(",")]

    metrics = ["SNR", "P5", "PE"] \
        + (["ViSQOL"] if args.visqol else []) + (["Zim"] if args.zim else [])
    print(f"bin:      {hqlc_bin}")
    print(f"clips:    {len(clips)} from {args.clips_dir}")
    print(f"bitrates: {', '.join(str(b) for b in bitrates)}")
    print(f"metrics:  {', '.join(metrics)}\n")

    tasks = [(br, c.stem, c, str(hqlc_bin), args.duration, args.visqol, args.zim)
             for br in bitrates for c in clips]
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for fut in as_completed(pool.submit(process_clip, t) for t in tasks):
            r = fut.result()
            rows.append(r)
            tag = f"  b{r['bitrate']//1000:>3d}k {r['clip']:<24s}"
            if r["status"] == "ok":
                extra = ""
                if args.visqol:
                    extra += f" V={r['visqol']:.3f}"
                if args.zim:
                    extra += f" Z={r['zim']:.3f}"
                print(f"{tag} SNR={r['snr']:6.2f} P5={r['seg_p5']:6.2f} "
                      f"PE={_fmt_pe(r['pe_worst'], r['n_attacks'])}{extra}", flush=True)
            else:
                print(f"{tag} {r['status']}", flush=True)

    # Summary: per bitrate — mean SNR/P5, worst P5, worst PE across clips.
    head = (f"\n{'bitrate':>7s}  {'SNR':>6s}  {'P5':>6s}  {'minP5':>6s}  {'maxPE':>6s}"
            + (f"  {'ViSQOL':>7s}" if args.visqol else "")
            + (f"  {'Zim':>7s}" if args.zim else "") + "   N")
    print(head)
    print("─" * (len(head) - 1))
    for br in bitrates:
        ok = [r for r in rows if r["bitrate"] == br and r["status"] == "ok"]
        if not ok:
            continue
        snr = sum(r["snr"] for r in ok) / len(ok)
        p5 = sum(r["seg_p5"] for r in ok) / len(ok)
        min_p5 = min(r["seg_p5"] for r in ok)
        pes = [r["pe_worst"] for r in ok if r["pe_worst"] is not None]
        max_pe = f"{max(pes):+6.1f}" if pes else "     —"
        line = f"{br:>7d}  {snr:>6.2f}  {p5:>6.2f}  {min_p5:>6.2f}  {max_pe}"
        if args.visqol:
            line += f"  {sum(r['visqol'] for r in ok) / len(ok):>7.3f}"
        if args.zim:
            line += f"  {sum(r['zim'] for r in ok) / len(ok):>7.3f}"
        line += f"  {len(ok):>3d}"
        print(line)

    if args.output:
        fields = ["bitrate", "clip", "snr", "seg_p5",
                  "pe_worst", "pe_mean", "n_attacks", "visqol", "zim", "status"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: (x["bitrate"], x["clip"])):
                w.writerow(r)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
