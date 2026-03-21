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

This produces `hqlc_enc`, a simple encode/decode tool:

```
hqlc_enc input.wav output.wav -b 96000    # rate-controlled at 96 kbps
hqlc_enc input.wav output.wav -g 2.0      # fixed gain
```

The public API is in [`include/hqlc.h`](include/hqlc.h) and should be self-descriptive. See [`src/hqlc_enc.c`](src/hqlc_enc.c) for a complete usage example. For ESP-IDF integration, there's a sample component and linker script in `benchmark/esp-bench/components/hqlc`.

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

The encoder needs ~15 KB of RAM (3 KB state + 12 KB scratch) and the decoder ~16 KB (4 KB state + 12 KB scratch). The compiled library is about 27 KB on ESP32 (`-Os`), keeping the overall footprint small enough for memory-constrained targets.

### Audio quality (ViSQOL MOS)

All codecs at 96 kbps stereo 48 kHz, scored with ViSQOL, higher is better.

**MUSDB18** (50 tracks, real mixed-style music):

| Codec | Mean | Min | Max |
|-------|------|-----|-----|
| HQLC | 4.615 | 4.442 | 4.718 |
| LC3 | 4.522 | 4.051 | 4.705 |
| Opus | 4.506 | 4.111 | 4.711 |
| AAC | 4.499 | 4.193 | 4.710 |
| MP3 | 4.107 | 3.390 | 4.724 |

**SQAM** (70 tracks, harder recordings):

| Codec | Mean | Min | Max |
|-------|------|-----|-----|
| HQLC | 4.633 | 4.093 | 4.732 |
| Opus | 4.570 | 4.061 | 4.732 |
| LC3 | 4.516 | 3.972 | 4.732 |
| AAC | 4.437 | 3.671 | 4.732 |
| MP3 | 4.210 | 3.082 | 4.732 |

HQLC scores highest on both datasets, but take this with a grain of salt. ViSQOL is an objective metric and likely doesn't fully capture artifacts like pre-echo, where codecs with more sophisticated psychoacoustic models (like Opus) may do better in practice. With that said, the scores do suggest that HQLC is quite competetive there. I should probably prepare an ABX test.

## Test clips

The `test-clips/` directory contains short audio clips under open licenses, intended as freely redistributable replacements for the hard-case tracks from SQAM (which can't be included in the repo). These are used for CI regression tests with ViSQOL.

- `bass_guitar.wav` — Serolillo, [CC BY 2.5](https://creativecommons.org/licenses/by/2.5), via Wikimedia Commons
- `flamenco_percussion.wav` — from [Freesound](https://freesound.org)
- Remaining clips — classical recordings from [Musopen](https://musopen.org) (public domain)

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
