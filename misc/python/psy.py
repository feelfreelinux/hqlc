"""
Psychoacoustic analysis, calculates the per-band exponent indices
"""

import math

import numpy as np

from .constants import (
    BAND_EDGES,
    EXP_INDEX_BIAS,
    EXP_INDEX_MAX,
    EXP_INDEX_MIN,
    FB_COARSE,
    FINE_BAND_EDGES,
    FINE_TILT_DB,
    N_ACTIVE_FINE,
    N_BANDS,
)


def tilt_for_bitrate(bitrate):
    """Pre-emphasis tilt in dB for a bitrate (C psy_tilt_for_bitrate).

    35 dB at >=128 kbps, ramps down 5 dB per 32 kbps, floor at 15 dB.
    """
    if bitrate >= 128000:
        return 35
    tilt = 35 - (128000 - bitrate) * 5 // 32000
    return max(15, tilt)


def tilt_step_q7(tilt_db):
    """Per-fine-band tilt increment in EXP_Q7 (128 per exponent index unit)"""
    return (int(tilt_db) * 118612 + 32768) >> 16


def compute_exponents(X, tilt_db=FINE_TILT_DB, transient=False):
    """Compute 20 coarse-band exponent indices from the MDCT spectrum.

    For non-transient / steady frames, we use apply the 1-2-1 PSD smoothing, and do the "hat-basis" aggregation
    (so the fine bands are used to better model the signal, this is the same idea behind interpolation in the quantizer)

    Transient frames use a plain geometric mean for the coarse bands. This is due to the fact that transients are sharp by definition, we do not want to model it's finer structure, or smooth its attack.
    """
    edges = FINE_BAND_EDGES

    # Per-fine-band mean energy (PSD)
    psd = [0.0] * N_ACTIVE_FINE
    for fb in range(N_ACTIVE_FINE):
        s, e = edges[fb], edges[fb + 1]
        psd[fb] = float((X[s:e] ** 2).sum()) / float(e - s)

    if not transient:
        # 1-2-1 low-pass across fine-band PSDs (linear domain, edges replicated)
        prev = psd[0]
        for fb in range(N_ACTIVE_FINE):
            nxt = psd[fb + 1] if fb + 1 < N_ACTIVE_FINE else psd[fb]
            sm = (prev + 2.0 * psd[fb] + nxt) / 4.0
            prev = psd[fb]
            psd[fb] = sm

    # Tilt increment per fine band, in exponent-index units
    tilt_per_fb = tilt_step_q7(tilt_db) / 128.0

    if not transient:
        # Hat-basis: each fine band's tilted log-PSD is split between its two
        # nearest coarse centers, weighted by distance. wl/ws accumulate the
        # weighted log-sum and total weight per coarse band. exp = wl/ws.
        centers = [(BAND_EDGES[b] + BAND_EDGES[b + 1] + 2) >> 1 for b in range(N_BANDS)]
        wl = [0.0] * N_BANDS
        ws = [0.0] * N_BANDS
        tilt_acc = 0.0
        k = 0
        exp = np.zeros(N_BANDS, dtype=np.int32)
        for fb in range(N_ACTIVE_FINE):
            lg = (2.0 * math.log2(psd[fb]) if psd[fb] > 0.0 else 0.0) + tilt_acc
            tilt_acc += tilt_per_fb
            x = (edges[fb] + edges[fb + 1]) >> 1  # fine-band center bin
            if x <= centers[0]:
                wl[0] += lg
                ws[0] += 1.0
            elif x >= centers[-1]:
                wl[-1] += lg
                ws[-1] += 1.0
            else:
                while x >= centers[k + 1]:
                    k += 1
                t = (x - centers[k]) / float(centers[k + 1] - centers[k])
                wl[k] += (1.0 - t) * lg
                ws[k] += 1.0 - t
                wl[k + 1] += t * lg
                ws[k + 1] += t
        for b in range(N_BANDS):
            v = wl[b] / ws[b] if ws[b] > 0.0 else 0.0
            exp[b] = int(
                np.clip(
                    math.floor(v + EXP_INDEX_BIAS + 0.5),
                    EXP_INDEX_MIN,
                    EXP_INDEX_MAX,
                )
            )
        return exp

    # Transient path - simple geometric mean per coarse band
    log_sum = [0.0] * N_BANDS
    cnt = [0] * N_BANDS
    tilt_acc = 0.0
    for fb in range(N_ACTIVE_FINE):
        b = FB_COARSE[fb]
        log_idx = 2.0 * math.log2(psd[fb]) if psd[fb] > 0.0 else 0.0
        log_sum[b] += log_idx + tilt_acc
        cnt[b] += 1
        tilt_acc += tilt_per_fb

    exp = np.zeros(N_BANDS, dtype=np.int32)
    for b in range(N_BANDS):
        if cnt[b] > 0:
            exp[b] = int(
                np.clip(
                    math.floor(log_sum[b] / cnt[b] + EXP_INDEX_BIAS + 0.5),
                    EXP_INDEX_MIN,
                    EXP_INDEX_MAX,
                )
            )
    return exp
