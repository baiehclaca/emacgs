/**
 * Pure-JavaScript implementation of Laplace-kernel attention.
 *
 * The same math as the PyTorch version in laplace_transformer/attention.py,
 * but written so the whole pattern-matching demo can run in the browser
 * with zero dependencies and zero backend.
 *
 * Key choice for a zero-training demo: we use one-hot character embeddings
 * so that the L1 distance between "the same char" is 0 and between any two
 * different chars is 2. The Laplace kernel  exp(-dist / b)  then gives a
 * clean, interpretable per-position match score.
 *
 * Exported as both a global (window.Laplace) and a CommonJS module for
 * simple Node-based tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.Laplace = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Map each unique character in `text` to an integer index. */
  function buildVocab(text) {
    const vocab = Object.create(null);
    let next = 0;
    for (const ch of text) {
      if (vocab[ch] === undefined) vocab[ch] = next++;
    }
    return vocab;
  }

  /**
   * Apply Laplace-kernel attention using one-hot char embeddings.
   *
   * @param {number[]} qIdx   Query char indices, length T_q.
   * @param {number[]} kIdx   Key char indices,   length T_k.
   * @param {number}   bandwidth   Positive scale `b`.
   * @param {number[]} [qPos]  Optional query positions.
   * @param {number[]} [kPos]  Optional key positions.
   * @param {number}   [positionAlpha=0]  Weight on position distance.
   * @returns {number[][]} weights[T_q][T_k], rows sum to 1.
   */
  function attention(qIdx, kIdx, bandwidth, qPos, kPos, positionAlpha) {
    const b = Math.max(bandwidth, 1e-6);
    const alpha = positionAlpha || 0;
    const Tq = qIdx.length;
    const Tk = kIdx.length;
    const weights = new Array(Tq);
    for (let i = 0; i < Tq; i++) {
      const row = new Array(Tk);
      // For numerical stability, subtract the max log-weight from each row.
      let maxLog = -Infinity;
      const logs = new Array(Tk);
      for (let j = 0; j < Tk; j++) {
        const charDist = qIdx[i] === kIdx[j] ? 0 : 2;
        const posDist = qPos && kPos ? Math.abs(qPos[i] - kPos[j]) : 0;
        const dist = charDist + alpha * posDist;
        const logw = -dist / b;
        logs[j] = logw;
        if (logw > maxLog) maxLog = logw;
      }
      let sum = 0;
      for (let j = 0; j < Tk; j++) {
        const e = Math.exp(logs[j] - maxLog);
        row[j] = e;
        sum += e;
      }
      for (let j = 0; j < Tk; j++) row[j] /= sum;
      weights[i] = row;
    }
    return weights;
  }

  /**
   * Score every position of `haystack` by how strongly the `needle` attends
   * to it through the Laplace kernel. Returns per-haystack-position totals
   * plus the full T_n × T_h attention matrix (handy for a heatmap).
   */
  function matchPattern(haystack, needle, bandwidth) {
    bandwidth = bandwidth === undefined ? 0.5 : bandwidth;
    const vocab = buildVocab(haystack + needle);
    const hIdx = Array.from(haystack, (c) => vocab[c]);
    const nIdx = Array.from(needle, (c) => vocab[c]);
    const weights = attention(nIdx, hIdx, bandwidth);
    const scores = new Array(haystack.length).fill(0);
    for (let i = 0; i < needle.length; i++) {
      for (let j = 0; j < haystack.length; j++) {
        scores[j] += weights[i][j];
      }
    }
    return { scores, weights, haystack, needle };
  }

  /**
   * Classic induction-head continuation.
   *
   * For each new token, attend from the last emitted char to every past
   * position, using the Laplace kernel as the match score, and pick the
   * character that most-often came *right after* a strong match.
   *
   * This is a zero-training analogue of the "induction heads" circuit in
   * real transformers: find previous occurrences of the current context,
   * copy what came next.
   */
  function inductionContinue(text, numTokens, bandwidth) {
    numTokens = numTokens || 8;
    bandwidth = bandwidth === undefined ? 0.3 : bandwidth;
    let out = text;
    const b = Math.max(bandwidth, 1e-6);

    for (let step = 0; step < numTokens; step++) {
      if (out.length < 2) break;
      const last = out[out.length - 1];
      const prev = out.length >= 2 ? out[out.length - 2] : null;
      const votes = Object.create(null);
      let total = 0;
      // Score every position i in [0, out.length - 2]; the candidate
      // emission is out[i + 1].
      for (let i = 0; i < out.length - 1; i++) {
        // 2-gram match: compare both (prev, last) and (out[i-1], out[i]).
        const d1 = out[i] === last ? 0 : 2;
        let d = d1;
        if (prev !== null && i > 0) {
          const d0 = out[i - 1] === prev ? 0 : 2;
          d += d0;
        }
        const w = Math.exp(-d / b);
        const next = out[i + 1];
        votes[next] = (votes[next] || 0) + w;
        total += w;
      }
      if (total === 0) break;
      // argmax
      let bestChar = null;
      let bestWeight = -1;
      for (const k in votes) {
        if (votes[k] > bestWeight) {
          bestWeight = votes[k];
          bestChar = k;
        }
      }
      if (bestChar === null) break;
      out += bestChar;
    }
    return out.slice(text.length);
  }

  return {
    buildVocab: buildVocab,
    attention: attention,
    matchPattern: matchPattern,
    inductionContinue: inductionContinue,
  };
});
