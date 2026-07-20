import { state } from "./state.js";

export const $ = (s) => document.querySelector(s);
export const $$ = (s) => document.querySelectorAll(s);

export function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Show the right step / phase
export function setPhase(phase) {
  $("#step-file").hidden = phase !== "setup";
  $("#step-config").hidden = phase !== "setup" || state.files.length === 0;
  $("#step-abx").hidden = phase !== "abx";
  $("#step-results").hidden = phase !== "results";
}
