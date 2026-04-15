(() => {
  const $ = (id) => document.getElementById(id);

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  }

  // ---------------------- Pattern match ---------------------------
  const matchBtn = $("matchBtn");
  const matchResult = $("matchResult");

  matchBtn.addEventListener("click", async () => {
    const haystack = $("haystack").value.trim();
    const needle = $("needle").value.trim();
    if (!haystack || !needle) {
      matchResult.innerHTML =
        '<div class="status bad">Please fill in both fields.</div>';
      return;
    }
    matchBtn.disabled = true;
    matchResult.innerHTML = '<div class="status">Scoring...</div>';
    try {
      const data = await postJSON("/api/match", { haystack, needle });
      renderScores(data);
    } catch (err) {
      matchResult.innerHTML =
        `<div class="status bad">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      matchBtn.disabled = false;
    }
  });

  function renderScores(data) {
    const { haystack, scores, needle } = data;
    if (!scores.length) {
      matchResult.innerHTML =
        '<div class="status">No scores returned.</div>';
      return;
    }
    const maxScore = Math.max(...scores);
    const bars = [];
    for (let i = 0; i < haystack.length; i++) {
      const s = scores[i] ?? 0;
      const pct = maxScore > 0 ? (s / maxScore) * 100 : 0;
      bars.push(`
        <div class="bar-wrap" title="pos ${i}: ${s.toFixed(4)}">
          <div class="bar-char">${escapeHtml(showChar(haystack[i]))}</div>
          <div class="bar-box"><div class="bar-fill" style="height:${pct.toFixed(
            1
          )}%"></div></div>
          <div class="bar-score">${(s * 100).toFixed(0)}</div>
        </div>
      `);
    }
    matchResult.innerHTML = `
      <div class="status good">
        needle: <strong>${escapeHtml(needle)}</strong>
         · bars = attention mass from needle → position
      </div>
      <div class="bars">${bars.join("")}</div>
    `;
  }

  // ---------------------- Copy task -------------------------------
  const copyBtn = $("copyBtn");
  const copyResult = $("copyResult");
  copyBtn.addEventListener("click", async () => {
    const pattern = $("pattern").value.trim();
    if (!pattern) {
      copyResult.innerHTML =
        '<div class="status bad">Type a pattern first.</div>';
      return;
    }
    copyBtn.disabled = true;
    copyResult.innerHTML = '<div class="status">Generating...</div>';
    try {
      const data = await postJSON("/api/copy", { pattern });
      const outCls = data.correct ? "good" : "bad";
      const statusCls = data.correct ? "good" : "bad";
      const statusText = data.correct
        ? "Correct copy"
        : "Model tried its best — try a shorter / letters-only pattern.";
      copyResult.innerHTML = `
        <div class="copy-line">
          <span class="prompt">${escapeHtml(data.prompt)}</span><span class="out ${outCls}">${escapeHtml(
        data.completion
      )}</span>
        </div>
        <div class="status ${statusCls}">${statusText}</div>
      `;
    } catch (err) {
      copyResult.innerHTML =
        `<div class="status bad">Error: ${escapeHtml(err.message)}</div>`;
    } finally {
      copyBtn.disabled = false;
    }
  });

  // ---------------------- helpers ---------------------------------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[c]));
  }
  function showChar(c) {
    if (c === " ") return "␣";
    return c;
  }

  // default values so first tap Just Works on phone
  $("haystack").value = "xxxabcdefxxxabcabc";
  $("needle").value = "abc";
  $("pattern").value = "quickfox";
})();
