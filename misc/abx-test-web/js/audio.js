import { state } from "./state.js";

// Decode each picked file to interleaved int16 PCM at 48kHz and store on state.
// Returns an array of error strings for files that failed to decode.
export async function loadFiles(fileList) {
  const files = [];
  const errors = [];
  for (const file of fileList) {
    try {
      const arrayBuf = await file.arrayBuffer();
      const ctx = new OfflineAudioContext(2, 1, 48000);
      const audioBuf = await ctx.decodeAudioData(arrayBuf);
      const decoded = await resampleTo48k(audioBuf);
      const channels = Math.min(decoded.numberOfChannels, 2);
      const nSamples = decoded.length;

      const pcm = new Int16Array(nSamples * channels);
      for (let ch = 0; ch < channels; ch++) {
        const f32 = decoded.getChannelData(ch);
        for (let i = 0; i < nSamples; i++) {
          pcm[i * channels + ch] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
        }
      }
      files.push({ name: file.name, channels, pcm });
    } catch (err) {
      errors.push(`Failed to decode ${file.name}: ${err.message}`);
    }
  }
  state.files = files;
  return errors;
}

async function resampleTo48k(audioBuf) {
  if (audioBuf.sampleRate === 48000) return audioBuf;
  const ch = Math.min(audioBuf.numberOfChannels, 2);
  const outLen = Math.round(audioBuf.length * (48000 / audioBuf.sampleRate));
  const offCtx = new OfflineAudioContext(ch, outLen, 48000);
  const src = offCtx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(offCtx.destination);
  src.start();
  return await offCtx.startRendering();
}
