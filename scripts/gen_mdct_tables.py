#!/usr/bin/env python3
"""Generate MDCT lookup tables (KBD window + twiddle factors).

Outputs: include/mdct_tables.h, src/mdct_tables.c
"""

import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MDCT_N = 512
FFT_N = 256
KBD_ALPHA = 3.0
Q31 = 2**31 - 1


def to_q31(x):
    return np.clip(np.round(x * Q31), -2**31, Q31).astype(np.int64)


def make_kbd_window(N, alpha=KBD_ALPHA):
    beta = np.pi * alpha
    w = np.i0(beta * np.sqrt(1.0 - ((2.0 * np.arange(N + 1) / N) - 1.0) ** 2)) / np.i0(beta)
    W = np.cumsum(w)
    kbd = np.zeros(2 * N)
    kbd[:N] = np.sqrt(W[:N] / W[N])
    kbd[N:] = kbd[N - 1::-1]
    return kbd


def make_twiddle(N, angle_fn):
    """Generate interleaved re/im twiddle array."""
    n = N // 2
    k = np.arange(n, dtype=np.float64)
    angle = angle_fn(k, N)
    out = np.empty(2 * n)
    out[0::2] = np.cos(angle)
    out[1::2] = -np.sin(angle)
    return out


def fmt_q31(name, values):
    q = to_q31(values)
    lines = [f"const int32_t {name}[{len(q)}] = {{"]
    for i in range(0, len(q), 4):
        chunk = q[i:i+4]
        entries = ", ".join(f"{int(v):11d}" for v in chunk)
        lines.append(f"    {entries}{',' if i + 4 < len(q) else ''}")
    lines.append("};")
    return "\n".join(lines)


def main():
    kbd = make_kbd_window(MDCT_N)
    kbd_half = kbd[:MDCT_N]
    assert np.allclose(kbd[:MDCT_N], kbd[2*MDCT_N-1:MDCT_N-1:-1]), "KBD not symmetric"

    pre = make_twiddle(MDCT_N, lambda k, N: np.pi * (4*k + 1) / (4*N))
    post = make_twiddle(MDCT_N, lambda k, N: np.pi * k / N)
    fft = make_twiddle(FFT_N, lambda k, N: 2 * np.pi * k / N)

    # Digit-reversal swap pairs for 256-point radix-4 FFT
    digit_rev = []
    for i in range(FFT_N):
        j = ((i & 0x03) << 6) | ((i & 0x0C) << 2) | ((i & 0x30) >> 2) | ((i & 0xC0) >> 6)
        if i < j:
            digit_rev.extend([i, j])

    src = os.path.join(PROJECT_DIR, "src", "mdct_tables.c")
    with open(src, "w") as f:
        f.write('#include "mdct_tables.h"\n\n')
        for name, data in [("kbd_window_half_q31", kbd_half),
                           ("lut_pre_twiddle_q31", pre),
                           ("lut_post_twiddle_q31", post),
                           ("lut_fft_twiddle_q31", fft)]:
            f.write(fmt_q31(name, data) + "\n\n")

        # Digit-reversal LUT
        lines = [f"const uint8_t lut_digit_rev[{len(digit_rev)}] = {{"]
        for i in range(0, len(digit_rev), 12):
            chunk = digit_rev[i:i+12]
            entries = ", ".join(f"{v:3d}" for v in chunk)
            lines.append(f"    {entries}{',' if i + 12 < len(digit_rev) else ''}")
        lines.append("};")
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {src}")


if __name__ == "__main__":
    main()
