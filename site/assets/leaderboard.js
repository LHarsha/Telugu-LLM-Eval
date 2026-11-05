(function () {
  "use strict";

  const GROUPS = {
    category: { label: "Capability", keys: ["cultural", "code_mixed", "reading_comprehension"],
      names: { cultural: "Cultural", code_mixed: "Code-mixed", reading_comprehension: "Reading comp." } },
    script: { label: "Script", keys: ["telugu", "mixed", "romanized"],
      names: { telugu: "Telugu script", mixed: "Mixed script", romanized: "Romanized" } },
    difficulty: { label: "Difficulty", keys: ["easy", "medium", "hard"],
      names: { easy: "Easy", medium: "Medium", hard: "Hard" } },
  };
  let data = null;
  let group = "category";
  let mode = "strict";

  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(1));

  function seg(el, options, current, onPick) {
    el.innerHTML = "";
    for (const [value, label] of options) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("aria-pressed", String(value === current));
      b.onclick = () => onPick(value);
      el.appendChild(b);
    }
  }

  function overallOf(m) {
    return mode === "judge" && m.judge ? m.judge.overall : m.overall;
  }

  function ciCell(o, chance) {
    if (!o || o.accuracy === null || o.accuracy === undefined) return "–";
    const ci = o.ci95 || [o.accuracy, o.accuracy];
    const L = (x) => `${(100 * x).toFixed(2)}%`;
    const chanceMark = chance != null ? `<span class="chance" style="left:${L(chance)}" title="random guess ${pct(chance)}%"></span>` : "";
    return `<div class="score">
      <div class="ci" role="img" aria-label="${pct(o.accuracy)}%, 95% CI ${pct(ci[0])} to ${pct(ci[1])}">
        <span class="range" style="left:${L(ci[0])};width:${L(Math.max(ci[1] - ci[0], 0.004))}"></span>
        <span class="pt" style="left:calc(${L(o.accuracy)} - 1px)"></span>${chanceMark}
      </div>
      <span class="num">${pct(o.accuracy)}<span class="sub">${pct(ci[0])}–${pct(ci[1])}</span></span>
    </div>`;
  }

  function render() {
    const g = GROUPS[group];
    const thead = $("#board thead");
    thead.innerHTML = `<tr><th>#</th><th>Model</th><th>Overall</th>${g.keys.map((k) => `<th>${esc(g.names[k])}</th>`).join("")}<th>Items</th></tr>`;
    const tbody = $("#board tbody");
    if (!data.models.length) {
      tbody.innerHTML = `<tr><td class="empty" colspan="${4 + g.keys.length}">No results yet.</td></tr>`;
      return;
    }
    const chance = data.dataset.chance_accuracy;
    let rank = 0;
    tbody.innerHTML = data.models.map((m) => {
      const ranked = m.complete && m.kind !== "baseline";
      if (ranked) rank += 1;
      const cls = [m.kind === "baseline" ? "baseline" : "", m.complete ? "" : "incomplete"].join(" ");
      const tag = m.kind === "open-weights" ? '<span class="tag open">open weights</span>'
        : m.kind === "baseline" ? '<span class="tag">baseline</span>' : '<span class="tag">API</span>';
      const status = m.complete ? "" : `<span class="sub">incomplete: ${m.n_scored}/${m.n_items} items answered so far</span>`;
      const groups = (m.groups && m.groups[group]) || {};
      const cells = g.keys.map((k) => {
        const v = groups[k];
        return `<td>${v ? `${pct(v.accuracy)}<span class="sub">n=${v.n}</span>` : "–"}</td>`;
      }).join("");
      const pf = m.parse_failures ? `<span class="sub">${m.parse_failures} unparseable</span>` : "";
      const selfJ = mode === "judge" && m.judge && m.judge.self_judged ? '<span class="sub">judged by itself</span>' : "";
      return `<tr class="${cls}">
        <td class="rank">${ranked ? rank : ""}</td>
        <td class="model"><b>${esc(m.display_name)}</b>${tag}${status}${selfJ}</td>
        <td class="overall">${ciCell(overallOf(m), chance)}</td>
        ${cells}
        <td>${m.n_scored}${pf}</td>
      </tr>`;
    }).join("");
  }

  function renderControls() {
    seg($("#groupSeg"), Object.entries(GROUPS).map(([k, v]) => [k, v.label]), group, (v) => { group = v; renderControls(); render(); });
    const hasJudge = data.judge && data.models.some((m) => m.judge);
    $("#judgeWrap").hidden = !hasJudge;
    if (hasJudge) {
      seg($("#judgeSeg"), [["strict", "Strict match"], ["judge", "With LLM judge"]], mode, (v) => { mode = v; renderControls(); render(); });
    }
    $("#legend").innerHTML = $("#legend").innerHTML.replace(/ Judge:.*$/, "") +
      (mode === "judge" && data.judge ? ` Judge: short answers that failed strict match were re-checked by ${esc(data.judge.display_name)}. Category columns always show strict scores.` : "");
  }

  function renderIntro() {
    const ds = data.dataset;
    // Only models that answered every item count as "evaluated".
    const nModels = data.models.filter((m) => m.kind !== "baseline" && m.complete).length;
    $("#facts").innerHTML = [
      [ds.n_items, "questions"],
      [3, "capabilities"],
      [nModels, nModels === 1 ? "model evaluated" : "models evaluated"],
      [ds.chance_accuracy != null ? pct(ds.chance_accuracy) + "%" : "–", "random-guess level"],
    ].map(([b, s]) => `<div><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
    const note = data.notice !== undefined ? data.notice : (data.split !== "reviewed" ? data.split_note : "");
    $("#notice").innerHTML = note ? `<p class="notice">${esc(note)}</p>` : "";
    $("#meta").textContent = `Dataset ${data.dataset_version} · prompt ${data.prompt_version} · updated ${data.generated}`;
  }

  fetch("data/results.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((d) => { data = d; renderIntro(); renderControls(); render(); })
    .catch(() => { $("#board tbody").innerHTML = '<tr><td class="empty" colspan="6">Results are not available yet.</td></tr>'; });
})();
