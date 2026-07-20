import { $, $$ } from "./dom.js";

// A, B and X share one clock so switching swaps the source at the current
// position instead of restarting. A short gain ramp on every start/stop keeps
// cutting a source from clicking.
const SWITCH_FADE = 0.006;

export const Transport = {
  ctx: null,
  bufs: { a: null, b: null, x: null },
  role: "x",
  source: null,
  gain: null,
  resumeAtPos: false, // false restarts the fragment on switch, true keeps position
  playing: false,
  startedAt: 0,
  offset: 0,
  duration: 0,

  ensureCtx() {
    if (!this.ctx) this.ctx = new AudioContext({ sampleRate: 48000 });
    if (this.ctx.state === "suspended") this.ctx.resume();
  },

  setTrial(a, b, x) {
    this._stopSource();
    this.bufs = { a, b, x };
    this.duration = a.duration;
    this.role = "x";
    this.offset = 0;
    this.playing = false;
    this._syncUi();
  },

  positionNow() {
    if (!this.playing) return this.offset;
    let pos = this.offset + (this.ctx.currentTime - this.startedAt);
    if (pos > this.duration) pos = this.duration;
    return pos;
  },

  _stopSource(fade = false) {
    const src = this.source;
    if (!src) return;
    src.onended = null;
    if (fade && this.gain) {
      const t = this.ctx.currentTime;
      try {
        this.gain.gain.cancelScheduledValues(t);
        this.gain.gain.setValueAtTime(this.gain.gain.value, t);
        this.gain.gain.linearRampToValueAtTime(0, t + SWITCH_FADE);
      } catch (e) {}
      try { src.stop(t + SWITCH_FADE + 0.002); } catch (e) {}
    } else {
      try { src.stop(); } catch (e) {}
    }
    this.source = null;
    this.gain = null;
  },

  _startAt(pos) {
    this._stopSource(true);
    const t = this.ctx.currentTime;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(1, t + SWITCH_FADE);
    g.connect(this.ctx.destination);
    const src = this.ctx.createBufferSource();
    src.buffer = this.bufs[this.role];
    src.connect(g);
    src.start(t, pos);
    src.onended = () => {
      if (this.source === src) {
        this.playing = false;
        this.offset = 0;
        this._syncUi();
      }
    };
    this.source = src;
    this.gain = g;
    this.startedAt = t;
    this.offset = pos;
    this.playing = true;
    this._syncUi();
  },

  switchTo(role) {
    const pos = this.resumeAtPos ? this.positionNow() : 0;
    this.role = role;
    this._startAt(pos);
  },

  toggle() {
    if (this.playing) this.pause();
    else this._startAt(this.offset >= this.duration ? 0 : this.offset);
  },

  pause() {
    this.offset = this.positionNow();
    this._stopSource(true);
    this.playing = false;
    this._syncUi();
  },

  seekFrac(f) {
    const pos = Math.max(0, Math.min(0.999, f)) * this.duration;
    if (this.playing) { this.offset = pos; this._startAt(pos); }
    else { this.offset = pos; this._syncUi(); }
  },

  _syncUi() {
    $("#btn-play").textContent = this.playing ? "Pause" : "Play";
    $$(".play-btn").forEach((b) => b.classList.toggle("active", b.dataset.role === this.role));
    renderTransport();
  },
};

export function buildBuffer(int16, channels) {
  const nSamples = int16.length / channels;
  const buf = Transport.ctx.createBuffer(channels, nSamples, 48000);
  for (let ch = 0; ch < channels; ch++) {
    const cd = buf.getChannelData(ch);
    for (let i = 0; i < nSamples; i++) cd[i] = int16[i * channels + ch] / 32768;
  }
  return buf;
}

function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function renderTransport() {
  const pos = Transport.duration ? Transport.positionNow() : 0;
  const frac = Transport.duration ? pos / Transport.duration : 0;
  $("#seek-fill").style.width = (frac * 100).toFixed(1) + "%";
  $("#time").textContent = `${fmtTime(pos)} / ${fmtTime(Transport.duration)}`;
}

// Keep the position readout live while playing.
(function raf() {
  if (!$("#step-abx").hidden && Transport.playing) renderTransport();
  requestAnimationFrame(raf);
})();
