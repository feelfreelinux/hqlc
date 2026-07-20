# HQLC - High Quality, Low Complexity audio codec

A low-complexity MDCT audio codec targeting 96 kbps+ stereo at 48 kHz, designed to run comfortably on embedded targets like the ESP32 while staying competitive with established codecs in audio quality.

All building blocks are public domain or based on expired patents. The codec supports 48 kHz sample rate only, with PCM16 or PCM24 input, at a fixed frame size of 512 samples (~10.67 ms). 

**See [HQLC_DESIGN.md](HQLC_DESIGN.md) for a detailed write-up of the codec internals and design rationale. This readme only roughly covers the build instructions & some benchmarks.**

## Building

The repository contains a fixed-point C implementation, unit tests, and a Python reference.

```
mkdir build && cd build
cmake .. -DHQLC_TOOLS=ON -GNinja && ninja
```

This produces `hqlc`, the CLI tool:

```
hqlc input.wav output.wav -b 96000         # roundtrip at 96 kbps (for ABX testing)
hqlc -e input.wav output.hqlc -b 128000    # encode to bitstream
hqlc -d output.hqlc decoded.wav            # decode bitstream to WAV
ffmpeg -i in.flac -f wav - | hqlc - out.wav  # pipe from ffmpeg
```

The public API is in [`include/hqlc.h`](include/hqlc.h) and should be self-descriptive. See [`src/hqlc_cli.c`](src/hqlc_cli.c) for a complete usage example. For ESP-IDF integration, there's a sample component and linker script in `benchmark/esp-bench/components/hqlc`.

## Python reference

A pure Python/NumPy reference implementation lives in `misc/python/`. It can be run directly from the repo root:

```
python -m misc.python input.wav output.wav -b 96000
```

## Benchmarks

### ESP32 encode/decode speed

Measured on an ESP32 at 240 MHz, stereo 48 kHz, ~2.1s of audio. Other codecs measured via `espressif/esp_audio_codec@2.4.1`.

| Codec | kbps | Enc (ms) | Dec (ms) |
|-------|------|----------|----------|
| SBC | 357 | 202 | 161 |
| HQLC | 96 | 344 | 327 |
| AAC | 129 | 1318 | 621 |
| Opus (c1) | 96 | 1379 | 855 |
| Opus (c5) | 96 | 1787 | 917 |
| LC3 | 96 | 2195 | 710 |

SBC is included as a baseline - it's a subband codec, hence simpler and faster than any MDCT-based design, but at the cost of much worse compression efficiency (357 kbps vs 96 kbps).

The current implementation has no SIMD/NEON optimizations since ESP32 was the primary target, but the codec's straightforward structure (radix-4 FFT, lattice filters, integer rANS) should make vectorization easy on other platforms.

### Memory and code size

The encoder needs ~13.5 KB of RAM (~3.2 KB state + ~10.2 KB scratch) and the decoder ~10 KB (~2 KB state + ~8.2 KB scratch). The compiled library is about 27 KB on ESP32 (`-Os`), keeping the overall footprint small enough for memory-constrained targets.

### Audio quality (ViSQOL + Zimtohrli MOS)

All codecs at 96 kbps stereo 48 kHz, scored with ViSQOL and Zimtohrli. Both are MOS-like objective metrics, higher is better. Rows are sorted by ViSQOL mean.

**MUSDB18** (50 tracks, real mixed-style music):

| Codec | ViSQOL | (min–max) | Zim | (min–max) |
|-------|--------|-----------|-----|-----------|
| HQLC | 4.613 | 4.367–4.725 | 4.732 | 4.578–4.881 |
| LC3 | 4.522 | 4.051–4.705 | 4.751 | 4.638–4.867 |
| Opus | 4.506 | 4.111–4.711 | 4.850 | 4.755–4.930 |
| AAC | 4.439 | 4.082–4.726 | 4.804 | 4.612–4.946 |
| MP3 | 4.107 | 3.390–4.724 | 4.412 | 3.729–4.904 |

**SQAM** (70 tracks, harder recordings):

| Codec | ViSQOL | (min–max) | Zim | (min–max) |
|-------|--------|-----------|-----|-----------|
| HQLC | 4.650 | 4.221–4.732 | 4.847 | 4.379–4.999 |
| Opus | 4.570 | 4.061–4.732 | 4.895 | 4.642–5.000 |
| LC3 | 4.516 | 3.972–4.732 | 4.778 | 2.893–4.995 |
| AAC | 4.410 | 3.637–4.732 | 4.882 | 4.688–5.000 |
| MP3 | 4.210 | 3.082–4.732 | 4.690 | 4.032–5.000 |

## Test clips

The `test-clips/` directory contains short audio clips under open licenses, intended as freely redistributable replacements for the hard-case tracks from SQAM (which can't be included in the repo). These are used for CI regression tests with ViSQOL.

- `bass_guitar.wav`: Serolillo, [CC BY 2.5](https://creativecommons.org/licenses/by/2.5), via Wikimedia Commons
- `flamenco_percussion.wav`: from [Freesound](https://freesound.org)
- Remaining clips: classical recordings from [Musopen](https://musopen.org) (public domain)

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
