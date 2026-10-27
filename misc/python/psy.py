"""
Psychoacoustic analysis, calculates the per-band exponent indices
"""

import math

import numpy as np

from .constants import (
    BAND_EDGES,
    DB_PER_EXP_INDEX,
    EXP_INDEX_BIAS,
    EXP_INDEX_MAX,
    EXP_INDEX_MIN,
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


def tilt_step(tilt_db):
    """Per-fine-band tilt increment, in exponent-index units.

    The tilt is spread evenly over the active fine bands, at DB_PER_EXP_INDEX
    per exponent index, rounded to Q7 like the C psy_tilt_step_q7 (step = tilt_db * 1.81)
    """
    return round(tilt_db / (N_ACTIVE_FINE * DB_PER_EXP_INDEX) * 128) / 128


_LOG_FLOOR = -400.0  # 2*log2 of an empty band: drives the exponent to the floor


def fine_band_psd(X):
    """Per-fine-band mean energy (PSD), 1-2-1 smoothed across bands with the edges
    replicated (C psy_fine_band_psd + psy_fine_band_psd_smooth)"""
    edges = FINE_BAND_EDGES
    psd = [
        float((X[edges[fb] : edges[fb + 1]] ** 2).sum())
        / float(edges[fb + 1] - edges[fb])
        for fb in range(N_ACTIVE_FINE)
    ]

    prev = psd[0]
    for fb in range(N_ACTIVE_FINE):
        nxt = psd[fb + 1] if fb + 1 < N_ACTIVE_FINE else psd[fb]
        sm = (prev + 2.0 * psd[fb] + nxt) / 4.0
        prev = psd[fb]
        psd[fb] = sm
    return psd


def compute_exponents(X, tilt_db=FINE_TILT_DB):
    """Compute 20 coarse-band exponent indices from the MDCT spectrum.

    See the C code for context
    """
    edges = FINE_BAND_EDGES
    psd = fine_band_psd(X)

    # Tilt increment per fine band, in exponent-index units
    tilt_per_fb = tilt_step(tilt_db)

    # each fine band's tilted log-PSD is split between its two earest coarse centers, weighted by distance.
    centers = [(BAND_EDGES[b] + BAND_EDGES[b + 1] + 2) >> 1 for b in range(N_BANDS)]
    wl = [0.0] * N_BANDS
    ws = [0.0] * N_BANDS
    tilt_acc = 0.0
    k = 0
    exp = np.zeros(N_BANDS, dtype=np.int32)
    for fb in range(N_ACTIVE_FINE):
        # psd==0 must go to the FLOOR, not lg=0 to match python
        lg = (2.0 * math.log2(psd[fb]) if psd[fb] > 0.0 else _LOG_FLOOR) + tilt_acc
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
