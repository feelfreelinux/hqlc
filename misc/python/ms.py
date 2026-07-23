"""Per-band M/S stereo coding, implements the same logic as ms.c"""

import math

from .constants import BAND_EDGES, EXP_INDEX_MAX, N_BANDS

# include/ms.h constants
MS_BIAS_GAP = 2
MS_BIAS_CAP = 6
MS_RANS_ALPHA_SHIFT = 2

# Gate thresholds in dB (C uses side/mid energy ratios 1/4 and 13/32).
MS_GATE_ENTER_DB = -6.0
MS_GATE_RESET_DB = -4.0


def ms_gate(XL, XR, ms_flags):
    """Per-band Schmitt gate on side/mid energy. Mirrors ms.c ms_gate().

    Updates ms_flags in place (persistent across frames = the hysteresis state);
    returns True if any band is flagged.
    """
    any_ms = False
    for b in range(N_BANDS):
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        mid = 0.5 * (XL[s:e] + XR[s:e])
        side = 0.5 * (XL[s:e] - XR[s:e])
        mid_e = float((mid * mid).sum())
        side_e = float((side * side).sum())

        if mid_e == 0.0:
            ms_flags[b] = False
        else:
            ratio_db = 10.0 * math.log10((side_e + 1e-20) / mid_e)
            if ratio_db < MS_GATE_ENTER_DB:
                ms_flags[b] = True
            elif ratio_db > MS_GATE_RESET_DB:
                ms_flags[b] = False

        any_ms = any_ms or ms_flags[b]
    return any_ms


def ms_encode(XL, XR, ms_flags):
    """Replace flagged L/R bands with M/S in place. Mirrors ms.c ms_encode()."""
    if not ms_gate(XL, XR, ms_flags):
        return
    for b in range(N_BANDS):
        if not ms_flags[b]:
            continue
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        mid = 0.5 * (XL[s:e] + XR[s:e])
        side = 0.5 * (XL[s:e] - XR[s:e])
        XL[s:e] = mid  # M
        XR[s:e] = side  # S


def ms_decode(X0, X1, ms_flags):
    """Replace flagged M/S bands with L/R in place."""
    for b in range(N_BANDS):
        if not ms_flags[b]:
            continue
        s, e = BAND_EDGES[b], BAND_EDGES[b + 1]
        mid = X0[s:e].copy()
        side = X1[s:e].copy()
        X0[s:e] = mid + side  # L
        X1[s:e] = mid - side  # R


def ms_apply_side_exp_bias(exp0, exp1, ms_flags):
    """Coarsen flagged S-band exponents"""
    for b in range(N_BANDS):
        if not ms_flags[b]:
            continue
        side_bias = int(exp0[b]) - int(exp1[b]) - MS_BIAS_GAP
        if side_bias > MS_BIAS_CAP:
            side_bias = MS_BIAS_CAP
        if side_bias > 0:
            exp1[b] = min(EXP_INDEX_MAX, int(exp1[b]) + side_bias)
