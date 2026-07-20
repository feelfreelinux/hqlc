import { state } from "./state.js";
import { $, $$, escapeHtml, setPhase } from "./dom.js";
import { config } from "./config.js";
import { initCodecs, codecsReady, roundtrip, label } from "./codecs.js";
import { loadFiles } from "./audio.js";
import { Transport, buildBuffer } from "./transport.js";
import { showResults } from "./results.js";
import { initSubmit, prepareSubmit } from "./submit.js";

// Whether to show the running score during the test
let showLiveScore = false;

initCodecs().catch((err) => showError(err.message));
initSubmit();
setPhase("setup");

// Show which codec build this deploy was compiled from (blank on local dev).
if (config.version) {
  const footer = $("#app-footer");
  footer.textContent = `HQLC codec build: ${config.version}`;
  footer.hidden = false;
}

function showError(msg) {
  const el = $("#load-error");
  el.textContent = msg;
  el.hidden = false;
}

// ---- File loading ----
$("#file-input").addEventListener("change", async (e) => {
  const fileList = e.target.files;
  if (!fileList.length) return;
  $("#load-error").hidden = true;

  const listEl = $("#file-list");
  listEl.innerHTML = "<p>Loading files...</p>";

  const errors = await loadFiles(fileList);
  for (const err of errors) showError(err);

  if (state.files.length === 0) {
    listEl.innerHTML = "";
    return;
  }

  listEl.innerHTML = state.files
    .map((f) => {
      const dur = (f.pcm.length / f.channels / 48000).toFixed(1);
      return `<div class="file-item">${escapeHtml(f.name)} (${dur}s, ${f.channels}ch)</div>`;
    })
    .join("");

  setPhase("setup"); // reveals config now that files exist
  updateTrialPlan();
});

// ---- Config ----
function selectedCodecs() {
  return Array.from($$(".codec-checks input:checked")).map((el) => el.value);
}

function updateTrialPlan() {
  const nCod = selectedCodecs().length;
  const nFiles = state.files.length;
  const per = parseInt($("#inp-trials").value, 10) || 0;
  const total = nCod * nFiles * per;
  $("#trial-plan").textContent =
    nFiles && nCod && per
      ? `= ${nCod} codec${nCod > 1 ? "s" : ""} x ${nFiles} track${nFiles > 1 ? "s" : ""} x ${per} = ${total} trials total`
      : "";
}
$("#inp-trials").addEventListener("input", updateTrialPlan);
$$(".codec-checks input").forEach((el) => el.addEventListener("change", updateTrialPlan));

// ---- Start test: encode every (codec, track), then build the trial list ----
$("#btn-start").addEventListener("click", async () => {
  if (!codecsReady()) return showError("WASM module not loaded yet.");
  const codecs = selectedCodecs();
  if (codecs.length === 0) return showError("Select at least one codec.");
  $("#load-error").hidden = true;

  const bitrate = parseInt($("#sel-bitrate").value, 10);
  state.trialsPerTrack = Math.max(1, parseInt($("#inp-trials").value, 10) || 1);

  $("#btn-start").disabled = true;
  $("#processing").hidden = false;

  const totalJobs = codecs.length * state.files.length;
  let doneJobs = 0;
  const fill = $("#progress-fill");
  fill.classList.remove("done");
  fill.style.width = "0%";

  state.codecData = {};
  for (const codec of codecs) {
    state.codecData[codec] = [];
    for (let fi = 0; fi < state.files.length; fi++) {
      $("#processing-text").textContent = `Encoding ${label(codec)} (${doneJobs + 1}/${totalJobs})`;
      $("#processing-detail").textContent = state.files[fi].name;
      fill.style.width = ((doneJobs / totalJobs) * 100).toFixed(1) + "%";
      await new Promise((r) => setTimeout(r, 20)); // yield to UI

      const result = roundtrip(codec, state.files[fi].pcm, state.files[fi].channels, bitrate);
      if (!result) {
        $("#processing").hidden = true;
        $("#btn-start").disabled = false;
        return showError(`${label(codec)} failed on ${state.files[fi].name}`);
      }
      state.codecData[codec].push(result);
      doneJobs++;
    }
  }

  fill.style.width = "100%";
  fill.classList.add("done");
  $("#processing-text").textContent = "Ready";
  $("#processing-detail").textContent = `${totalJobs} encodes complete: ${codecs.map(label).join(", ")}`;
  await new Promise((r) => setTimeout(r, 400));

  // Trim originals and encoded output to the shortest per file (codec latency).
  for (let fi = 0; fi < state.files.length; fi++) {
    let minLen = state.files[fi].pcm.length;
    for (const codec of codecs) minLen = Math.min(minLen, state.codecData[codec][fi].length);
    state.files[fi].pcm = state.files[fi].pcm.subarray(0, minLen);
    for (const codec of codecs) state.codecData[codec][fi] = state.codecData[codec][fi].subarray(0, minLen);
  }

  // Balanced trial list: every (codec, track) gets trialsPerTrack trials, shuffled.
  state.trials = [];
  for (const codec of codecs) {
    for (let fi = 0; fi < state.files.length; fi++) {
      for (let t = 0; t < state.trialsPerTrack; t++) {
        state.trials.push({ fileIdx: fi, codec, xIsA: Math.random() < 0.5, answer: null, correct: false });
      }
    }
  }
  for (let i = state.trials.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [state.trials[i], state.trials[j]] = [state.trials[j], state.trials[i]];
  }

  Transport.ensureCtx();
  Transport.resumeAtPos = $("#chk-resume").checked;
  showLiveScore = $("#chk-livescore").checked;
  $("#processing").hidden = true;
  $("#btn-start").disabled = false;
  startAbx();
});

// ---- ABX flow ----
function startAbx() {
  state.currentTrial = 0;
  $("#trial-tot").textContent = state.trials.length;
  const history = $("#history");
  history.innerHTML = "";
  for (let i = 0; i < state.trials.length; i++) {
    const dot = document.createElement("div");
    dot.className = "dot pending";
    history.appendChild(dot);
  }
  $("#score-live").hidden = !showLiveScore;
  setPhase("abx");
  loadTrial();
}

function loadTrial() {
  if (state.currentTrial >= state.trials.length) return finishTest();
  const t = state.trials[state.currentTrial];
  const f = state.files[t.fileIdx];
  const dur = (f.pcm.length / f.channels / 48000).toFixed(1);

  $("#trial-cur").textContent = state.currentTrial + 1;
  $("#trial-file").textContent = `${f.name} (${dur}s)`;

  const a = buildBuffer(f.pcm, f.channels);
  const b = buildBuffer(state.codecData[t.codec][t.fileIdx], f.channels);
  Transport.setTrial(a, b, t.xIsA ? a : b);

  const dots = $$("#history .dot");
  dots.forEach((d) => d.classList.remove("current"));
  if (dots[state.currentTrial]) dots[state.currentTrial].classList.add("current");
  updateLiveScore();
}

function updateLiveScore() {
  if (!showLiveScore) return;
  const answered = state.trials.slice(0, state.currentTrial);
  const correct = answered.filter((t) => t.correct).length;
  $("#score-live").textContent = `${correct} / ${answered.length} correct`;
}

function answer(ans) {
  if (state.currentTrial >= state.trials.length) return;
  const t = state.trials[state.currentTrial];
  if (t.answer !== null) return;
  t.answer = ans;
  t.correct = (ans === "a" && t.xIsA) || (ans === "b" && !t.xIsA);

  const dots = $$("#history .dot");
  dots[state.currentTrial].className = showLiveScore
    ? "dot " + (t.correct ? "correct" : "wrong")
    : "dot answered";

  state.currentTrial++;
  Transport.pause();
  loadTrial();
}

function finishTest() {
  Transport.pause();
  showResults();
  prepareSubmit();
}

// ---- Wiring ----
$$(".play-btn").forEach((btn) => btn.addEventListener("click", () => Transport.switchTo(btn.dataset.role)));
$$(".ans-btn").forEach((btn) => btn.addEventListener("click", () => answer(btn.dataset.ans)));
$("#btn-play").addEventListener("click", () => Transport.toggle());
$("#seek").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  Transport.seekFrac((e.clientX - r.left) / r.width);
});
$("#btn-abort").addEventListener("click", finishTest);

$("#btn-copy").addEventListener("click", async () => {
  const text = $("#report").textContent;
  const btn = $("#btn-copy");
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e2) {}
    document.body.removeChild(ta);
  }
  const old = btn.textContent;
  btn.textContent = "Copied";
  setTimeout(() => (btn.textContent = old), 1500);
});

$("#btn-again").addEventListener("click", () => {
  setPhase("setup");
  updateTrialPlan();
});
$("#btn-newfiles").addEventListener("click", () => {
  state.files = [];
  $("#file-input").value = "";
  $("#file-list").innerHTML = "";
  setPhase("setup");
});
