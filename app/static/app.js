const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const UNV = "[Not backed by manuals] ";
const EXAMPLES = [
  "Performance degradation observed over time",
  "Excessive vibration during operation",
  "Abnormal temperature rise in the unit",
  "Unusual noise from rotating components",
];
const STAGES = [
  ["investigator", "Investigator", "Retrieve evidence"],
  ["planner", "Planner", "Draft check plan"],
  ["verifier", "Verifier", "Check claims"],
  ["report", "Report", "Compile output"],
];

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}
async function poll(jobId, onUpdate) {
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    onUpdate(job);
    if (job.status !== "running") return job;
    await sleep(1000);
  }
}

/* ---------- status LEDs + library ---------- */
async function refreshHealth() {
  try {
    const h = await api("/api/health");
    $("#led-llm").className = "led " + (h.llm_ready ? "on" : "off");
    $("#led-emb").className = "led " + (h.embeddings_ready ? "on" : "off");
    $("#led-idx").className = "led " + (h.chunks > 0 ? "on" : "off");
    $("#idx-count").textContent = `INDEX ${h.chunks}`;
  } catch { /* server unreachable */ }
}
async function refreshDocs() {
  const { documents } = await api("/api/documents");
  $("#doclist").innerHTML = documents.length
    ? documents.map((d) => `<li class="doc"><b>${esc(d.name)}</b>
        <button class="x" data-name="${esc(d.name)}" title="Delete">x</button>
        <small>${d.chunks ? `<span class="tag ok">INDEXED</span>${d.chunks} chunks` : `<span class="tag no">NOT INDEXED</span>`} / ${d.size_kb} KB</small></li>`).join("")
    : `<li class="hint">No documents yet.</li>`;
  refreshHealth();
}
$("#doclist").addEventListener("click", async (e) => {
  const name = e.target.dataset?.name;
  if (!name || !confirm(`Delete ${name} and its index entries?`)) return;
  try { await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" }); } catch (err) { alert(err.message); }
  refreshDocs();
});

/* ---------- upload ---------- */
async function upload(fileList) {
  const form = new FormData();
  [...fileList].forEach((f) => form.append("files", f));
  const bar = $("#upload-bar"), msg = $("#upload-msg");
  msg.textContent = "Uploading...";
  try {
    const { job_id } = await api("/api/upload", { method: "POST", body: form });
    bar.hidden = false;
    const job = await poll(job_id, (j) => {
      bar.firstElementChild.style.width = `${Math.round(j.progress * 100)}%`;
      msg.textContent = j.stage || "Working...";
    });
    msg.textContent = job.status === "done" ? "Indexing complete." : `Error: ${job.error}`;
  } catch (err) { msg.textContent = `Error: ${err.message}`; }
  setTimeout(() => { bar.hidden = true; bar.firstElementChild.style.width = "0"; }, 1500);
  refreshDocs();
}
const dz = $("#dropzone"), fi = $("#file-input");
fi.addEventListener("change", () => fi.files.length && upload(fi.files));
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => e.dataTransfer.files.length && upload(e.dataTransfer.files));

/* ---------- pipeline ---------- */
function renderPipeline(job) {
  const done = new Set(job ? job.steps : []);
  $("#pipeline").innerHTML = STAGES.map(([id, name, desc], i) => {
    let cls = done.has(id) ? "done" : "idle";
    if (job && job.status === "running" && job.active === id) cls = "active";
    const rev = id === "planner" && job && job.revisions > 0 ? `<em class="rev">REV ${job.revisions}</em>` : "";
    return `<div class="stage ${cls}"><span class="stage-no">0${i + 1}</span><strong>${name}${rev}</strong><small>${desc}</small></div>`;
  }).join('<span class="wire"></span>');
}

/* ---------- diagnosis ---------- */
const cite = (src, page) => (src ? `<span class="cite">${esc(src)} &middot; p.${page}</span>` : `<span class="cite none">no manual citation</span>`);

function renderResult(r) {
  const banner = {
    verified: ["ok", "VERIFIED // all claims matched the retrieved passages"],
    partial: ["warn", "PARTIALLY VERIFIED // some claims are uncertain, see Uncertainty"],
    no_evidence: ["bad", "NO EVIDENCE // the uploaded documents do not cover this problem"],
  }[r.status];

  const findings = r.findings.length
    ? r.findings.map((f) => `<div class="card"><strong>${esc(f.cause)}</strong>
        <span class="meter l${{ high: 3, medium: 2, low: 1 }[f.confidence] || 1}" title="${esc(f.confidence)} confidence"><i></i><i></i><i></i></span>
        <p>${esc(f.evidence)}</p>${cite(f.source, f.page)}</div>`).join("")
    : `<p class="hint">No supported findings. Try rephrasing, or upload a more relevant manual.</p>`;

  const steps = r.plan.steps.length
    ? [...r.plan.steps].sort((a, b) => a.order - b.order).map((s) => {
        const unverified = s.action.startsWith(UNV);
        return `<div class="step"><div class="num">${s.order}</div><div>
          <div>${unverified ? '<span class="unv">UNVERIFIED</span>' : ""}${esc(unverified ? s.action.slice(UNV.length) : s.action)}</div>
          <p class="hint">${esc(s.rationale)}</p>${cite(s.source, s.page)}
          ${s.safety_note ? `<div class="warnline">WARNING: ${esc(s.safety_note)}</div>` : ""}</div></div>`;
      }).join("")
    : `<p class="hint">No checks can be suggested from the provided documents.</p>`;

  const unsupported = r.verification?.unsupported_claims || [];
  const uncertainty = unsupported.length
    ? `<ul class="plain">${unsupported.map((u) => `<li>${esc(u)}</li>`).join("")}</ul>`
    : `<p class="hint">No unsupported claims flagged. Retrieval may still have missed relevant pages.</p>`;

  const safety = [...new Set([...r.safety_notes, ...r.plan.safety_notes])];
  const passages = r.passages.map((p) => `<details><summary>${cite(p.source, p.page)} <span class="hint">relevance ${p.score}</span></summary><pre>${esc(p.text)}</pre></details>`).join("");

  $("#result").hidden = false;
  $("#result").innerHTML = `
    <div class="banner ${banner[0]}">${banner[1]}</div>
    <div class="result-grid">
      <div class="panel"><h2><em>A</em> Findings</h2>${findings}</div>
      <div class="panel"><h2><em>B</em> Suggested diagnostic checks</h2>${steps}</div>
      <div class="panel"><h2><em>C</em> Uncertainty</h2>${uncertainty}</div>
      <div class="panel safety"><div class="hz"></div><h2><em>D</em> Safety notes</h2><ul class="plain">${safety.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></div>
      <div class="panel"><h2><em>E</em> Sources consulted</h2>${passages || '<p class="hint">None.</p>'}</div>
      <div><a class="btn ghost" href="/reports/${encodeURIComponent(r.report_file)}" style="text-decoration:none;display:inline-block">Download report (.md)</a></div>
    </div>`;
  $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runDiagnosis() {
  const btn = $("#run"), err = $("#run-error");
  err.hidden = true;
  $("#result").hidden = true;
  btn.disabled = true;
  btn.textContent = "Running...";
  try {
    const { job_id } = await api("/api/diagnose", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: $("#query").value }),
    });
    const job = await poll(job_id, renderPipeline);
    if (job.status === "done") renderResult(job.result);
    else throw new Error(job.error);
  } catch (e) {
    err.textContent = `ERROR: ${e.message}`;
    err.hidden = false;
  }
  btn.disabled = false;
  btn.textContent = "Run diagnosis";
}

/* ---------- init ---------- */
$("#examples").innerHTML = EXAMPLES.map((t) => `<button type="button" class="chip">${esc(t)}</button>`).join("");
$("#examples").addEventListener("click", (e) => { if (e.target.classList.contains("chip")) { $("#query").value = e.target.textContent; $("#query").dispatchEvent(new Event("input")); } });
$("#query").addEventListener("input", () => ($("#count").textContent = `${$("#query").value.length} / 1000`));
$("#run").addEventListener("click", runDiagnosis);
renderPipeline(null);
refreshDocs();