// Minimal config.h for building Opus and LAME with Emscripten
#ifndef WASM_CONFIG_H
#define WASM_CONFIG_H

// Opus
#define OPUS_BUILD 1
#define HAVE_LRINTF 1
#define VAR_ARRAYS 1
#define FLOATING_POINT 1
#define PACKAGE_VERSION "1.6-wasm"

// LAME
#define HAVE_STDINT_H 1
#define HAVE_ERRNO_H 1
#define HAVE_STRCHR 1
#define HAVE_MEMCPY 1
#define STDC_HEADERS 1
#define LAME_LIBRARY_BUILD 1
#define ieee754_float32_t float

#endif
