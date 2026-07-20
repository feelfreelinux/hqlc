import { state } from "./state.js";
import { $ } from "./dom.js";
import { config } from "./config.js";
import { aggregate } from "./results.js";
import { Transport } from "./transport.js";

// When endpoint is empty the submit UI stays hidden and nothing is ever sent.
const enabled = () => typeof config.endpoint === "string" && /^https:\/\//.test(config.endpoint);

// Reveal and reset the submit block for a fresh set of results.
export function prepareSubmit() {
  const block = $("#submit-block");
  const { done } = aggregate();
  if (!enabled() || done.length === 0) {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  const btn = $("#btn-submit");
  btn.disabled = false;
  btn.textContent = "Submit results";
  const status = $("#submit-status");
  status.textContent = "";
  status.className = "submit-status";
}

function buildSubmission() {
  const bitrate = parseInt($("#sel-bitrate").value, 10) / 1000;
  const { done, codecs, perCodec, perTrack } = aggregate();
  return {
    token: config.token || "",
    codecVersion: config.version || "",
    submittedAt: new Date().toISOString(),
    nickname: ($("#inp-nick").value || "").slice(0, 40),
    comment: ($("#inp-comment").value || "").slice(0, 500),
    bitrateKbps: bitrate,
    trialsPerTrack: state.trialsPerTrack,
    codecs,
    resumeAtPos: !!Transport.resumeAtPos,
    userAgent: navigator.userAgent,
    perCodec,
    perTrack,
    trials: done.map((t) => ({
      track: state.files[t.fileIdx].name,
      codec: t.codec,
      xWas: t.xIsA ? "A" : "B",
      answer: t.answer,
      correct: t.correct,
    })),
    report: $("#report").textContent,
  };
}

export function initSubmit() {
  $("#btn-submit").addEventListener("click", async () => {
    if (!enabled()) return;
    const btn = $("#btn-submit");
    const status = $("#submit-status");
    btn.disabled = true;
    btn.textContent = "Submitting...";
    status.className = "submit-status";
    status.textContent = "";

    try {
      const res = await fetch(config.endpoint, {
        method: "POST",
        body: JSON.stringify(buildSubmission()),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let ok = true;
      try {
        const body = await res.json();
        ok = body.ok !== false;
      } catch (e) {
        // Non-JSON 200 counts as success.
      }
      if (!ok) throw new Error("rejected");
      btn.textContent = "Submitted";
      status.className = "submit-status ok";
      status.textContent = "Thanks, your results were recorded.";
    } catch (err) {
      btn.disabled = false;
      btn.textContent = "Submit results";
      status.className = "submit-status err";
      status.textContent = "Could not submit.";
    }
  });
}
