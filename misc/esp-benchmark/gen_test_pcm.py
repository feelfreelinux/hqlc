#!/usr/bin/env python3
"""Convert a 48 kHz stereo WAV to a C header with embedded int16 PCM data.

Usage: python3 gen_test_pcm.py <input.wav> <output.h> [max_frames] [skip_frames]

max_frames limits the clip length (each frame = 512 samples).
Default is 200 frames (~2.13 seconds), which keeps flash usage ~800 KB.
skip_frames skips that many frames from the start (default 0).
"""

import struct
import sys
import wave

FRAME_SAMPLES = 512


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.wav> <output.h> [max_frames]")
        sys.exit(1)

    wav_path = sys.argv[1]
    out_path = sys.argv[2]
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    skip_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 48000, f"Expected 48 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, f"Expected 16-bit, got {w.getsampwidth() * 8}-bit"
        n_ch = w.getnchannels()
        n_total = w.getnframes()

        # Skip initial frames
        skip_samples = skip_frames * FRAME_SAMPLES
        if skip_samples > 0:
            w.readframes(min(skip_samples, n_total))

        n_samples = n_total - skip_samples

        # Limit to max_frames worth of codec frames (need +1 for MDCT overlap)
        max_samples = (max_frames + 1) * FRAME_SAMPLES
        n_samples = min(n_samples, max_samples)

        raw = w.readframes(n_samples)

    samples = struct.unpack(f"<{n_samples * n_ch}h", raw)

    with open(out_path, "w") as f:
        f.write(f"/* Auto-generated from {wav_path.split('/')[-1]} */\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define TEST_PCM_CHANNELS {n_ch}\n")
        f.write(f"#define TEST_PCM_SAMPLES  {n_samples}\n")
        f.write(f"#define TEST_PCM_FRAMES   {(n_samples // FRAME_SAMPLES) - 1}\n\n")
        f.write(
            f"static const int16_t test_pcm[{n_samples * n_ch}] = {{\n"
        )
        for i in range(0, len(samples), 16):
            chunk = samples[i : i + 16]
            f.write("  " + ", ".join(f"{s}" for s in chunk) + ",\n")
        f.write("};\n")

    kb = len(samples) * 2 / 1024
    print(
        f"{wav_path.split('/')[-1]}: {n_ch}ch, {n_samples} samples, "
        f"{(n_samples // FRAME_SAMPLES) - 1} codec frames, {kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
