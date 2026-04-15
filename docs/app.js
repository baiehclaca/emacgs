/* UI glue for the Laplace-Transformer static demo.
   All computation runs locally via laplace.js — no backend. */
(() => {
  const $ = (id) => document.getElementById(id);
  const Laplace = window.Laplace;

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }
  const showChar = (c) => (c === " " ? "␣" : c);

  // ---------------- bandwidth slider ------------------------------------
  const bw = $("bw");
  const bwVal = $("bwVal");
  bw.addEventListener("input", () => {
    bwVal.textContent = Number(bw.value).toFixed(2);
    runMatch();    // live-update score bars
    runHeatmap();
  });

  // ---------------- continuation slider ---------------------------------
  const numTokens = $("numTokens");
  const nVal = $("nVal");
  numTokens.addEventListener("input", () => {
    nVal.textContent = numTokens.value;
  });

  // ---------------- pattern match ---------------------------------------
  function runMatch() {
    const haystack = $("haystack").value;
    const needle = $("needle").value;
    const result = $("matchResult");
    if (!haystack || !needle) {
      result.innerHTML = "";
      return;
    }
    try {
      const b = Number(bw.value);
      const { scores } = Laplace.matchPattern(haystack, needle, b);
      renderScores(result, haystack, needle, scores);
    } catch (err) {
      result.innerHTML =
        '<div class="status bad">Error: ' + escapeHtml(err.message) + "</div>";
    }
  }
  $("matchBtn").addEventListener("click", runMatch);
  $("haystack").addEventListener("input", () => {
    runMatch();
    runHeatmap();
  });
  $("needle").addEventListener("input", () => {
    runMatch();
    runHeatmap();
  });

  function renderScores(container, haystack, needle, scores) {
    const maxScore = scores.length ? Math.max.apply(null, scores) : 0;
    const bars = [];
    for (let i = 0; i < haystack.length; i++) {
      const s = scores[i] || 0;
      const pct = maxScore > 0 ? (s / maxScore) * 100 : 0;
      bars.push(
        '<div class="bar-wrap" title="pos ' + i + ": " + s.toFixed(4) + '">' +
          '<div class="bar-char">' + escapeHtml(showChar(haystack[i])) + "</div>" +
          '<div class="bar-box"><div class="bar-fill" style="height:' +
          pct.toFixed(1) + '%"></div></div>' +
          '<div class="bar-score">' + (s * 100).toFixed(0) + "</div>" +
        "</div>"
      );
    }
    container.innerHTML =
      '<div class="status good">needle: <strong>' +
      escapeHtml(needle) +
      "</strong> &middot; bars = attention mass from needle &rarr; position</div>" +
      '<div class="bars">' + bars.join("") + "</div>";
  }

  // ---------------- induction-head continuation -------------------------
  function runContinue() {
    const context = $("context").value;
    const n = Number(numTokens.value);
    const b = Number(bw.value);
    const result = $("contResult");
    if (!context) {
      result.innerHTML =
        '<div class="status bad">Type some context first.</div>';
      return;
    }
    try {
      const cont = Laplace.inductionContinue(context, n, Math.max(b * 0.6, 0.05));
      if (!cont) {
        result.innerHTML =
          '<div class="status bad">Context too short for induction.</div>';
        return;
      }
      result.innerHTML =
        '<div class="copy-line"><span class="prompt">' +
        escapeHtml(context) +
        '</span><span class="out">' +
        escapeHtml(cont) +
        "</span></div>";
    } catch (err) {
      result.innerHTML =
        '<div class="status bad">Error: ' + escapeHtml(err.message) + "</div>";
    }
  }
  $("contBtn").addEventListener("click", runContinue);
  $("context").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runContinue();
  });

  // ---------------- heatmap ---------------------------------------------
  function runHeatmap() {
    const haystack = $("haystack").value;
    const needle = $("needle").value;
    const el = $("heatmap");
    if (!haystack || !needle) {
      el.innerHTML = "";
      return;
    }
    const b = Number(bw.value);
    const { weights } = Laplace.matchPattern(haystack, needle, b);
    let maxW = 0;
    for (const row of weights) for (const v of row) if (v > maxW) maxW = v;

    const rows = [];
    // column header (haystack chars)
    const hdr = ['<td class="hdr"></td>'];
    for (const ch of haystack) {
      hdr.push('<td class="hdr">' + escapeHtml(showChar(ch)) + "</td>");
    }
    rows.push("<tr>" + hdr.join("") + "</tr>");

    for (let i = 0; i < needle.length; i++) {
      const tds = [
        '<td class="hdr">' + escapeHtml(showChar(needle[i])) + "</td>",
      ];
      for (let j = 0; j < haystack.length; j++) {
        const v = weights[i][j];
        const t = maxW > 0 ? v / maxW : 0;
        // dark-blue → light-blue gradient
        const alpha = 0.15 + 0.85 * t;
        tds.push(
          '<td style="background: rgba(122,162,255,' + alpha.toFixed(3) + ')" ' +
            'title="' + v.toFixed(4) + '"></td>'
        );
      }
      rows.push("<tr>" + tds.join("") + "</tr>");
    }
    el.innerHTML = "<table>" + rows.join("") + "</table>";
  }

  // ---------------- defaults + first render -----------------------------
  $("haystack").value = "xxxabcdefxxxabcabc";
  $("needle").value = "abc";
  $("context").value = "abcabcabc";
  runMatch();
  runHeatmap();
})();
