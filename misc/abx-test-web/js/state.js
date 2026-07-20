// Shared mutable state for the whole test. Modules import and mutate this.
export const state = {
  files: [],       // { name, channels, pcm: Int16Array (interleaved, 48kHz) }
  codecData: {},   // codecData[codec][fileIdx] = Int16Array
  trials: [],      // { fileIdx, codec, xIsA, answer, correct }
  trialsPerTrack: 5,
  currentTrial: 0,
};
