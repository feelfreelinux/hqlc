// One-tailed binomial p-value: P(X >= k) under H0 that the listener is guessing
// (p = 0.5). This is the standard significance test for a 2-alternative ABX run.
export function binomialPValue(k, n) {
  let p = 0;
  for (let i = k; i <= n; i++) p += binomialPMF(i, n);
  return Math.min(1, p);
}

function binomialPMF(k, n) {
  return Math.exp(logBinomCoeff(n, k) - n * Math.LN2);
}

function logBinomCoeff(n, k) {
  if (k > n) return -Infinity;
  let r = 0;
  for (let i = 0; i < k; i++) r += Math.log(n - i) - Math.log(i + 1);
  return r;
}
