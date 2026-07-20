import { state } from "./state.js";
import { $, $$, escapeHtml, setPhase } from "./dom.js";
import { config } from "./config.js";
import { binomialPValue } from "./stats.js";
import { label } from "./codecs.js";

// Roll the answered trials up per codec and per track.
export function aggregate() {
  const done = state.trials.filter((t) => t.answer !== null);
  const codecs = [...new Set(done.map((t) => t.codec))];
  const perCodec = {};
  const perTrack = {};
  for (const codec of codecs) {
    const ct = done.filter((t) => t.codec === codec);
    perCodec[codec] = { correct: ct.filter((t) => t.correct).length, total: ct.length };
    const byFile = {};
    for (const t of ct) {
      const name = state.files[t.fileIdx].name;
      (byFile[name] ||= { correct: 0, total: 0 });
      byFile[name].total++;
      if (t.correct) byFile[name].correct++;
    }
    perTrack[codec] = Object.entries(byFile).map(([name, v]) => ({ name, ...v }));
  }
  return { done, codecs, perCodec, perTrack };
}

export function showResults() {
  setPhase("results");

  const bitrate = parseInt($("#sel-bitrate").value, 10) / 1000;
  const { done, codecs, perCodec, perTrack } = aggregate();

  if (done.length === 0) {
    $("#results-by-codec").innerHTML = "<p>No trials answered.</p>";
    $("#results-per-track").innerHTML = "";
    $("#report").textContent = "";
    $("#results-table tbody").innerHTML = "";
    return;
  }

  let html = `<p class="results-bitrate">All codecs at ${bitrate} kbps</p>`;
  for (const codec of codecs) {
    const { correct, total } = perCodec[codec];
    const p = binomialPValue(correct, total);
    const sig = p < 0.05;
    html += `<div class="codec-result">
      <div class="codec-name">${label(codec)}</div>
      <div class="codec-score">${correct} / ${total}</div>
      <div class="codec-pval">p = ${p.toFixed(4)}</div>
      <div class="codec-verdict" style="color:${sig ? "var(--ok)" : "var(--muted)"}">
        ${sig ? "Distinguishable" : "Not distinguishable"}
      </div>
    </div>`;
  }
  $("#results-by-codec").innerHTML = html;

  let ptHtml = "<h3>Per track</h3>";
  for (const codec of codecs) {
    ptHtml += `<div class="track-group"><h4>${label(codec)}</h4><table class="track-table"><tbody>`;
    for (const row of perTrack[codec]) {
      const p = binomialPValue(row.correct, row.total);
      const sig = p < 0.05;
      ptHtml += `<tr>
        <td>${escapeHtml(row.name)}</td>
        <td class="right">${row.correct} / ${row.total}</td>
        <td class="right">p = ${p.toFixed(4)}</td>
        <td class="right" style="color:${sig ? "var(--ok)" : "var(--muted)"}">${sig ? "distinguishable" : "-"}</td>
      </tr>`;
    }
    ptHtml += "</tbody></table></div>";
  }
  $("#results-per-track").innerHTML = ptHtml;

  $("#report").textContent = buildReport(bitrate, codecs, perCodec, perTrack);

  const tbody = $("#results-table tbody");
  tbody.innerHTML = "";
  done.forEach((t, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${i + 1}</td>` +
      `<td>${escapeHtml(state.files[t.fileIdx].name)}</td>` +
      `<td>${label(t.codec)}</td>` +
      `<td>${t.xIsA ? "A" : "B"}</td>` +
      `<td>${t.answer.toUpperCase()}</td>` +
      `<td style="color:${t.correct ? "var(--ok)" : "var(--bad)"}">${t.correct ? "yes" : "no"}</td>`;
    tbody.appendChild(tr);
  });
}

export function buildReport(bitrate, codecs, perCodec, perTrack) {
  const date = new Date().toISOString().slice(0, 10);
  const L = [];
  L.push("HQLC ABX listening test");
  L.push(`Bitrate : ${bitrate} kbps  (48 kHz / 16-bit, lossless source)`);
  L.push("Method  : double-blind ABX, self-administered");
  L.push(`Trials  : ${state.trialsPerTrack} per track per codec`);
  L.push("Stats   : one-tailed binomial vs guessing (H0 p=0.5); * = p<0.05");
  L.push(`Date    : ${date}`);
  if (config.version) L.push(`Build   : HQLC ${config.version}`);
  L.push("");

  const codW = Math.max(6, ...codecs.map((c) => label(c).length));
  L.push("OVERALL");
  L.push(`  ${pad("codec", codW)}  ${pad("score", 8)}  ${pad("p-value", 9)}  verdict`);
  for (const codec of codecs) {
    const { correct, total } = perCodec[codec];
    const p = binomialPValue(correct, total);
    const star = p < 0.05 ? "*" : " ";
    L.push(
      `  ${pad(label(codec), codW)}  ${pad(`${correct}/${total}`, 8)}  ${pad(p.toFixed(4), 7)} ${star}  ${
        p < 0.05 ? "distinguishable" : "not distinguishable"
      }`
    );
  }

  L.push("");
  L.push("PER TRACK");
  const nameW = Math.min(34, Math.max(8, ...codecs.flatMap((c) => perTrack[c].map((r) => r.name.length))));
  for (const codec of codecs) {
    L.push(`  ${label(codec)}`);
    for (const row of perTrack[codec]) {
      const p = binomialPValue(row.correct, row.total);
      const star = p < 0.05 ? "*" : " ";
      L.push(`    ${pad(trunc(row.name, nameW), nameW)}  ${pad(`${row.correct}/${row.total}`, 6)}  ${pad(p.toFixed(4), 7)} ${star}`);
    }
  }
  return L.join("\n");
}

const pad = (s, n) => String(s).padEnd(n);
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 3) + "..." : s);
