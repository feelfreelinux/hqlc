# HQLC in-browser ABX test

A double-blind ABX listening test that roundtrips audio through HQLC (and optionally Opus and MP3) via WebAssembly.

## Build

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html)(`emcc` on your `PATH`). Opus and LAME are fetched and built by CMake.

```sh
emcmake cmake -B build -G Ninja
cmake --build build
cp build/abx_codecs.js build/abx_codecs.wasm .
```
