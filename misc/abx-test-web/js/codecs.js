// Codec roundtrip through the Emscripten module
export const CODEC_LABELS = { hqlc: "HQLC", opus: "Opus", mp3: "MP3" };
export const label = (c) => CODEC_LABELS[c] || c.toUpperCase();

let mod = null;

export async function initCodecs() {
  if (typeof window.createCodecModule === "undefined") {
    throw new Error("WASM module not found. Build it first (see README).");
  }
  mod = await window.createCodecModule();
}

export const codecsReady = () => mod !== null;

// Encode then decode interleaved int16 PCM, returning the roundtripped PCM (or
// null on failure). Bitrate is in bits per second.
export function roundtrip(codec, pcm, channels, bitrate) {
  const nSamples = pcm.length / channels;
  const inBytes = nSamples * channels * 2;
  const inPtr = mod._malloc(inBytes);
  const outPtr = mod._malloc(inBytes);
  mod.HEAP16.set(pcm, inPtr >> 1);

  const outSamples = mod.ccall(
    "roundtrip_" + codec,
    "number",
    ["number", "number", "number", "number", "number"],
    [inPtr, outPtr, nSamples, channels, bitrate]
  );

  let result = null;
  if (outSamples > 0) {
    result = new Int16Array(outSamples * channels);
    result.set(mod.HEAP16.subarray(outPtr >> 1, (outPtr >> 1) + outSamples * channels));
  }
  mod._free(inPtr);
  mod._free(outPtr);
  return result;
}
