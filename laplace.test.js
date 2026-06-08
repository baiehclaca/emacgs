/**
 * Tiny Node-based smoke test for laplace.js. Runs without any deps:
 *
 *     node docs/laplace.test.js
 */
"use strict";

const Laplace = require("./laplace.js");

let failed = 0;
function check(name, cond, detail) {
  if (cond) {
    console.log("  ok   " + name);
  } else {
    console.log("  FAIL " + name + (detail ? "   " + detail : ""));
    failed++;
  }
}

// --- attention rows sum to 1 --------------------------------------------
{
  const vocab = Laplace.buildVocab("abcabc");
  const idx = Array.from("abcabc", (c) => vocab[c]);
  const w = Laplace.attention(idx, idx, 0.5);
  for (let i = 0; i < w.length; i++) {
    const s = w[i].reduce((a, b) => a + b, 0);
    check("row " + i + " sums to 1", Math.abs(s - 1) < 1e-9, "sum=" + s);
  }
}

// --- pattern match finds repeated pattern -------------------------------
{
  const { scores, haystack } = Laplace.matchPattern(
    "xxxabcdefxxxabcabc",
    "abc",
    0.5
  );
  check("scores length matches haystack", scores.length === haystack.length);
  // Positions that start the pattern "abc" are 3, 12, 15; they should get
  // higher aggregate attention than filler "x" positions at 0, 1, 2.
  const patternAvg =
    (scores[3] + scores[4] + scores[5] + scores[12] + scores[13] + scores[14] +
      scores[15] + scores[16] + scores[17]) /
    9;
  const fillerAvg = (scores[0] + scores[1] + scores[2]) / 3;
  check(
    "pattern positions beat filler",
    patternAvg > fillerAvg * 1.2,
    "pattern=" + patternAvg.toFixed(4) + " filler=" + fillerAvg.toFixed(4)
  );
}

// --- induction head extends a repeating sequence ------------------------
{
  const out = Laplace.inductionContinue("abcabcabc", 3, 0.3);
  check(
    "induction extends abc repeat",
    out === "abc",
    "got=" + JSON.stringify(out)
  );
}

// --- bandwidth controls sharpness ---------------------------------------
{
  const tiny = Laplace.matchPattern("ababab", "a", 0.01);
  const wide = Laplace.matchPattern("ababab", "a", 5.0);
  // With a very small bandwidth the non-matching positions should be nearly
  // zero, so the max-to-min ratio is large. With a large bandwidth the
  // scores should be almost uniform.
  const tRatio =
    Math.max.apply(null, tiny.scores) / Math.min.apply(null, tiny.scores);
  const wRatio =
    Math.max.apply(null, wide.scores) / Math.min.apply(null, wide.scores);
  check(
    "small b is sharp, large b is flat",
    tRatio > wRatio * 10,
    "tiny=" + tRatio.toFixed(2) + " wide=" + wRatio.toFixed(4)
  );
}

if (failed > 0) {
  console.log("\n" + failed + " FAILED");
  process.exit(1);
}
console.log("\nall tests passed");
