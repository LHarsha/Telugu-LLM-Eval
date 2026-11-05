(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const LETTERS = "ABCDEF";
  const CATS = {
    cultural: ["Cultural knowledge", "Festivals, food, places, idioms, customs and history of Telangana and Andhra."],
    code_mixed: ["Code-mixed understanding", "Telugu–English mixing as people actually type it, in Telugu script and Romanized."],
    reading_comprehension: ["Reading comprehension", "A short Telugu passage followed by questions about it."],
  };
  let show = false;

  function seg() {
    const el = $("#ansSeg");
    el.innerHTML = "";
    for (const [v, label] of [[false, "Hidden"], [true, "Shown"]]) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label;
      b.setAttribute("aria-pressed", String(v === show));
      b.onclick = () => { show = v; document.body.classList.toggle("show-answers", show); seg(); };
      el.appendChild(b);
    }
  }

  function item(it) {
    const meta = `<div class="meta"><code>${esc(it.id)}</code><br>${esc(it.subtopic || "")}<br>${esc(it.script)} · ${esc(it.difficulty)}${it.region && it.region !== "none" ? "<br>" + esc(it.region) : ""}</div>`;
    const passage = it.passage ? `<details><summary>Passage</summary><div class="passage" lang="te">${esc(it.passage)}</div></details>` : "";
    let answer = "";
    if (it.answer_type === "mcq") {
      answer = `<ol class="opts">${(it.options || []).map((o, i) =>
        `<li class="${LETTERS[i] === it.gold_option ? "gold" : ""}"><b>${LETTERS[i]}</b><span lang="te">${esc(o)}</span></li>`).join("")}</ol>`;
    } else {
      answer = `<div class="gold-short">Accepted answers: <span lang="te">${(it.gold_answers || []).map(esc).join(" · ")}</span></div>`;
    }
    return `<div class="q">${meta}<div><div class="text" lang="te">${esc(it.question)}</div></div>${passage}${answer}</div>`;
  }

  fetch("data/samples.json", { cache: "no-cache" })
    .then((r) => r.json())
    .then((d) => {
      $("#cats").innerHTML = Object.entries(CATS).map(([k, [title, desc]]) => {
        const c = d.categories[k] || { items: [] };
        const note = c.source === "format_examples"
          ? '<p class="notice">Format examples only. These show the item format and are not part of the benchmark or its scores.</p>' : "";
        return `<div class="cat-head"><h2>${esc(title)}</h2><p>${esc(desc)}</p></div>${note}${c.items.map(item).join("") || '<p class="empty">No items yet.</p>'}`;
      }).join("");
    })
    .catch(() => { $("#cats").innerHTML = '<p class="empty">Examples are not available yet.</p>'; });
  seg();
})();
