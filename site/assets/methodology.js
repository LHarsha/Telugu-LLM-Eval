(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (o) => Object.entries(o || {}).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ");

  fetch("data/samples.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const p = d.prompts || {};
    $("#pMcq").textContent = p.mcq || "–";
    $("#pShort").textContent = p.short || "–";
    $("#pRc").textContent = p.reading_comprehension || "–";
    $("#pJudge").textContent = d.judge_prompt_template || "–";
  }).catch(() => {});

  fetch("data/results.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => {
    const ds = d.dataset;
    $("#dsStats").innerHTML = [
      ["Items", `${ds.n_items} (${d.split === "reviewed" ? "reviewed" : esc(d.split_note)})`],
      ["Capabilities", fmt(ds.by_category)],
      ["Script", fmt(ds.by_script)],
      ["Difficulty", fmt(ds.by_difficulty)],
      ["Format", fmt(ds.by_answer_type)],
      ["Passages", ds.n_passages],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("");
    if (d.judge) $("#judgeName").textContent = ` (currently ${d.judge.display_name})`;
    $("#meta").textContent = `Dataset ${d.dataset_version} · prompt ${d.prompt_version} (${d.template_hash}) · updated ${d.generated}`;
  }).catch(() => {});
})();
