/* openlab show — shared core for the design templates (demo data, plan semantics, a Word-style review annotator).
   Every template loads this file and builds its own layout around it. Nothing is sent anywhere: the data is made up.

   API (window.LAB):
     LAB.data                      sessions, plans (versions per agent), people (declared human workers, each with its own
                                  plans), proposal (a document), docComments, feed, inbox, queue
     LAB.STATES[state]             {label, fill, stroke, ink}  (planned, active, waiting, done, failed, dropped)
     LAB.EVIDENCE[ev]              {label, outline}            (ok: the record confirms, rep: the agent's word, bad: the record disagrees)
     LAB.plan(worker, i?)          a plan version by index (0 is v1; default the latest): {v, note, nodes:[{id,title,kind,needs,state,ev,evidence,owner}]};
                                  a worker is an agent or a person; a step's owner is the plan's worker unless `owner` names another
     LAB.layers(nodes)             {id: depth} for a left-to-right DAG layout
     LAB.diff(prev, cur)           {added:[titles], removed:[titles], changed:["title: state a → b; evidence; needs; renamed"]}
     LAB.graph(svg, nodes, opts)   draws a DAG into an <svg>; opts: {colW, nodeW, nodeH, gap, font, onClick(node), selected, prevIds, theme}
     LAB.renderDoc(docEl)          writes the proposal into docEl as a document (headings, paragraphs, lists, a table)
     LAB.annotate(docEl, marginEl, opts)
                                  Word-style review: select text → "Comment" → highlight + a margin card aligned to it,
                                  replies, resolve, delivery to the agent (shown as advice at its next tool call).
                                  opts: {author:"you", agent:"lead-7c0f", onChange(comments), button:"Comment"}
                                  Call it once per document (again returns the same instance) while the document is
                                  visible, or call .layout() when it is shown; it handles a CSS scale() around the page.
     LAB.spawn({role, prompt, usd, minutes, model, parent})  adds a session (demo) and returns it
     LAB.declare({id, role, cap, key, step})  declares a person who works in the plans beside the agents (demo)
     LAB.answer(id, {decision, message, picks, other})  decides an inbox item: approve/deny (a person's act, with an optional
                                  message to the agent) or the answer to an agent's question (picks and an "other" text)
     LAB.on(event, fn) / LAB.emit(event, payload)   tiny event bus: "feed", "sessions", "comments", "inbox"
*/
(function () {
  const LAB = (window.LAB = {});
  const esc = (t) => String(t == null ? "" : t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  LAB.esc = esc;
  const bus = {};
  LAB.on = (e, f) => (bus[e] = bus[e] || []).push(f);
  LAB.emit = (e, p) => (bus[e] || []).forEach((f) => f(p));

  LAB.STATES = {
    planned: { label: "planned", fill: "#eef1f4", stroke: "#8a96a3", ink: "#46505b" },
    active: { label: "active", fill: "#dbe9fb", stroke: "#2a78d6", ink: "#1c5cab" },
    waiting: { label: "waiting", fill: "#fff6e0", stroke: "#c98500", ink: "#8a5d00" },
    done: { label: "done", fill: "#e3f5e3", stroke: "#0ca30c", ink: "#0a6e0a" },
    failed: { label: "failed", fill: "#fbe4e4", stroke: "#d03b3b", ink: "#a52828" },
    dropped: { label: "dropped", fill: "#f4f5f7", stroke: "#b3bcc6", ink: "#626d78" },
  };
  LAB.EVIDENCE = {
    ok: { label: "the record confirms it", short: "on record", outline: "solid" },
    rep: { label: "the agent's word only", short: "agent's word", outline: "dashed" },
    bad: { label: "the record disagrees", short: "record disagrees", outline: "alarm" },
  };

  const N = (id, title, kind, needs, state, ev, evidence, owner) => ({ id, title, kind, needs, state, ev, evidence, ...(owner ? { owner } : {}) });
  const D = {};
  LAB.data = D;
  D.workspace = "~/research";
  D.box = { host: "gpu-box", running: 2, waiting: 1, gpus: "2× RTX 3090", until: "2026-10-20 08:00" };
  D.spend = 3.42;
  D.sessions = [
    { id: "lead-7c0f", role: "lead", state: "working", usd: 1.84, budget: 6, parent: null, study: "stations",
      q: "Do neural post-processing models beat EMOS on station temperature forecasts?", last: "agent_plan v3", ago: "40s" },
    { id: "exp-a1", role: "experimenter", state: "working", usd: 0.92, budget: 2, parent: "lead-7c0f", study: "stations",
      q: "Branch: DRN (distributional regression network)", last: "study_run --op improve --seed 2", ago: "12s" },
    { id: "exp-b2", role: "experimenter", state: "queued", usd: 0.31, budget: 2, parent: "lead-7c0f", study: "stations",
      q: "Branch: quantile regression forest", last: "queued: waiting for 2 GPUs", ago: "4m" },
    { id: "rev-c3", role: "reviewer", state: "idle", usd: 0.12, budget: 1, parent: "lead-7c0f", study: "stations",
      q: "Critique the DRN code before the selection", last: "study_critique: 3 findings", ago: "7m" },
    { id: "scr-d4", role: "screener", state: "done", usd: 0.23, budget: 0.5, parent: "lead-7c0f", study: "review-postproc",
      q: "Screen the literature (PRISMA)", last: "done: 214 screened, 31 included", ago: "1h" },
  ];
  D.plans = {
    "lead-7c0f": [
      { v: 1, note: "the first plan, from the approved proposal", nodes: [
        N("lit", "Review: post-processing methods", "review", [], "done", "ok", "prisma:91aa — 214 screened, 31 included"),
        N("data", "Station panel 2019–2024", "data", [], "done", "ok", "dataset:c84f — 2,192 rows, sha d743…"),
        N("emos", "Baseline: EMOS", "run", ["data"], "active", "rep", "the agent: fitting"),
        N("drn", "Branch: DRN", "branch", ["emos"], "planned", "rep", ""),
        N("ana", "Branch: analog ensemble", "branch", ["emos"], "planned", "rep", ""),
        N("t1", "Test DRN vs EMOS", "test", ["drn"], "planned", "rep", ""),
        N("sel", "Select the final candidate", "select", ["t1"], "planned", "rep", ""),
        N("look", "Holdout look (a person)", "person", ["sel"], "planned", "rep", "", "you"),
        N("paper", "Write-up and critique", "paper", ["look"], "planned", "rep", "")] },
      { v: 2, note: "analog ensemble dropped (a dead end); lagged features added", nodes: [
        N("lit", "Review: post-processing methods", "review", [], "done", "ok", "prisma:91aa — 214 screened, 31 included"),
        N("data", "Station panel 2019–2024", "data", [], "done", "ok", "dataset:c84f — 2,192 rows, sha d743…"),
        N("emos", "Baseline: EMOS", "run", ["data"], "done", "ok", "run:1a07 — audit clean · MAE 0.912 [0.88, 0.95]"),
        N("feat", "Lagged observations as features", "run", ["emos"], "done", "bad", "run:3c19 — the audit found LOOK-AHEAD at 2 of 20 steps; the agent says done, the record disagrees"),
        N("drn", "Branch: DRN", "branch", ["emos"], "active", "rep", "the agent: training; run:4fd2 in flight"),
        N("ana", "Branch: analog ensemble", "branch", ["emos"], "dropped", "ok", "dead_end:77e1 — comparison:c12 shows no gain, Δ −0.002 [−0.011, 0.006]"),
        N("t1", "Test DRN vs EMOS", "test", ["drn"], "planned", "rep", ""),
        N("sel", "Select the final candidate", "select", ["t1"], "planned", "rep", ""),
        N("look", "Holdout look (a person)", "person", ["sel"], "planned", "rep", "", "you"),
        N("paper", "Write-up and critique", "paper", ["look"], "planned", "rep", "")] },
      { v: 3, note: "a quantile-forest branch added; the selection waits for both tests and the critique", nodes: [
        N("lit", "Review: post-processing methods", "review", [], "done", "ok", "prisma:91aa — 214 screened, 31 included"),
        N("data", "Station panel 2019–2024", "data", [], "done", "ok", "dataset:c84f — 2,192 rows, sha d743…"),
        N("emos", "Baseline: EMOS", "run", ["data"], "done", "ok", "run:1a07 — audit clean · MAE 0.912 [0.88, 0.95]"),
        N("feat", "Lagged observations as features", "run", ["emos"], "failed", "bad", "run:3c19 — the audit found LOOK-AHEAD; a fix is planned (op debug)"),
        N("drn", "Branch: DRN", "branch", ["emos"], "done", "ok", "run:4fd2 — audit clean · MAE 0.861 [0.83, 0.89]"),
        N("qrf", "Branch: quantile forest", "branch", ["emos"], "waiting", "ok", "queue ticket 43 — waiting for 2 GPUs"),
        N("ana", "Branch: analog ensemble", "branch", ["emos"], "dropped", "ok", "dead_end:77e1 — no gain (comparison:c12)"),
        N("crit", "Critique of the DRN code", "critique", ["drn"], "active", "rep", "rev-c3 is reviewing"),
        N("t1", "Test DRN vs EMOS", "test", ["drn"], "done", "ok", "test:9b0e — p 0.003; alpha wealth 0.21 left"),
        N("t2", "Test QRF vs EMOS", "test", ["qrf"], "planned", "rep", ""),
        N("sel", "Select the final candidate (seed_robust)", "select", ["t1", "t2", "crit"], "planned", "rep", "rule sealed in the plan: seed_robust, k = 3"),
        N("look", "Holdout look (a person)", "person", ["sel"], "waiting", "ok", "inbox #7 — waits for you", "you"),
        N("paper", "Write-up and critique", "paper", ["look"], "planned", "rep", "")] },
    ],
    "exp-a1": [{ v: 1, note: "the DRN branch", nodes: [
      N("d1", "Draft: DRN, 2 layers", "run", [], "done", "ok", "run:2e11 — audit clean · MAE 0.889"),
      N("d2", "Improve: station embeddings", "run", ["d1"], "done", "ok", "run:4fd2 — MAE 0.861 · lesson:7 cited"),
      N("d3", "Seeds 2 and 3 of the best", "run", ["d2"], "active", "rep", "queue ticket 41 running"),
      N("d4", "Lesson: station embeddings help", "lesson", ["d2"], "done", "ok", "lesson:7 — improves; comparison:c15 Δ −0.028 [−0.041, −0.016]")] }],
    "exp-b2": [{ v: 1, note: "the quantile-forest branch", nodes: [
      N("q1", "Draft: QRF", "run", [], "waiting", "ok", "queue ticket 43 — waiting for 2 GPUs"),
      N("q2", "Test QRF vs EMOS", "test", ["q1"], "planned", "rep", "")] }],
    "rev-c3": [{ v: 1, note: "a review before the selection", nodes: [
      N("c1", "Read the DRN code", "review", [], "done", "rep", "an agent note"),
      N("c2", "File a critique", "critique", ["c1"], "done", "ok", "critique:k4 — 3 findings"),
      N("c3", "Answer findings with checks", "check", ["c2"], "planned", "rep", "")] }],
    "scr-d4": [{ v: 1, note: "screening", nodes: [
      N("s1", "Screen 214 sources", "review", [], "done", "ok", "screen nodes ×214; 3 conflicts sent to the inbox, resolved")] }],
  };
  // People who work in the plans beside the agents. A person is declared, not spawned: hours instead of a $ budget.
  // Their steps carry state and evidence the same way: a step a person reports is their word (dashed) until a record id
  // is linked, and a person's act (a look, an approval) still needs the key holder's signature in the inbox.
  D.people = [
    { id: "you", kind: "person", role: "principal investigator", state: "available", hours: 0, cap: 0, key: true,
      q: "Holds the person key: signs the looks, approvals and amendments", last: "approved the proposal", ago: "1h", plans: [] },
    { id: "sam", kind: "person", role: "meteorologist", state: "working", hours: 4.5, cap: 10, key: false,
      q: "Hand checks the agents cannot do: outage labels, metadata, a forecaster's read of the maps", last: "s-res: 2 of 5 regions read", ago: "6m",
      plans: [
        { v: 1, note: "hand checks for the station panel, at lead-7c0f's request", nodes: [
          N("s-lbl", "Label station outages by hand", "data", [], "done", "ok", "dataset:e2b1 — 412 outage labels; recipes/outages.py reads sam's sheet"),
          N("s-meta", "Check station metadata (moves, sensor swaps)", "check", [], "active", "rep", "sam: checking against the WMO station list"),
          N("s-res", "Read the DRN residual maps for physical sense", "review", ["s-meta"], "planned", "rep", "")] },
        { v: 2, note: "a forecaster's check of the QRF quantiles, since lead-7c0f's plan v3 added the branch", nodes: [
          N("s-lbl", "Label station outages by hand", "data", [], "done", "ok", "dataset:e2b1 — 412 outage labels; recipes/outages.py reads sam's sheet"),
          N("s-meta", "Check station metadata (moves, sensor swaps)", "check", [], "done", "rep", "sam: “checked against the WMO list, 3 moves found” — no record id linked"),
          N("s-res", "Read the DRN residual maps for physical sense", "review", ["s-meta"], "active", "rep", "sam: 2 of 5 regions read"),
          N("s-qrf", "Check the QRF quantiles against forecaster judgement", "review", ["s-res"], "planned", "rep", "")] }] },
    { id: "jo", kind: "person", role: "data engineer", state: "away", until: "Wed", hours: 1.5, cap: 4, key: false,
      q: "The ERA5 inputs and the data recipe for the write-up", last: "dataset:5a90 registered", ago: "1d",
      plans: [{ v: 1, note: "inputs for the station panel", nodes: [
        N("j-era", "Pull ERA5 2-m temperature for the panel", "data", [], "done", "ok", "dataset:5a90 — 2019–2024, sha 0e1c…"),
        N("j-doc", "Document the data recipes for the write-up", "paper", ["j-era"], "planned", "rep", "")] }] },
  ];
  LAB.plan = (w, v) => { const p = D.plans[w] || (D.people.find((x) => x.id === w) || {}).plans || []; return p.length ? p[v == null ? p.length - 1 : v] : null; };
  LAB.layers = (nodes) => {
    const by = Object.fromEntries(nodes.map((n) => [n.id, n])), memo = {};
    const depth = (id) => { if (memo[id] != null) return memo[id]; const ds = (by[id].needs || []).filter((x) => by[x]).map(depth);
      return (memo[id] = ds.length ? Math.max(...ds) + 1 : 0); };
    nodes.forEach((n) => depth(n.id)); return memo;
  };
  LAB.diff = (prev, cur) => {
    if (!prev) return { added: [], removed: [], changed: [] };
    const pid = new Set(prev.nodes.map((n) => n.id)), cid = new Set(cur.nodes.map((n) => n.id));
    return { added: cur.nodes.filter((n) => !pid.has(n.id)).map((n) => n.title),
      removed: prev.nodes.filter((n) => !cid.has(n.id)).map((n) => n.title),
      changed: cur.nodes.map((n) => { const o = prev.nodes.find((x) => x.id === n.id); if (!o) return null;
        const parts = [];
        if (o.title !== n.title) parts.push(`renamed from “${o.title}”`);
        if (o.state !== n.state) parts.push(`${o.state} → ${n.state}`);
        if (o.ev !== n.ev) parts.push(`${LAB.EVIDENCE[o.ev].short} → ${LAB.EVIDENCE[n.ev].short}`);
        const a = (o.needs || []).join(","), b = (n.needs || []).join(",");
        if (a !== b) parts.push(`needs ${(n.needs || []).join(", ") || "nothing"} (was ${(o.needs || []).join(", ") || "nothing"})`);
        if ((o.owner || "") !== (n.owner || "")) parts.push(`assigned to ${n.owner || "the plan's own worker"}`);
        return parts.length ? `${n.title}: ${parts.join("; ")}` : null; }).filter(Boolean) };
  };

  // --- a DAG renderer (templates may draw plans their own way instead) ---------------------------------------------
  const SVGNS = "http://www.w3.org/2000/svg";
  const el = (t, a, p, txt) => { const n = document.createElementNS(SVGNS, t); for (const k in a) n.setAttribute(k, a[k]); if (txt != null) n.textContent = txt; if (p) p.appendChild(n); return n; };
  LAB.svg = el;
  LAB.graph = (svg, nodes, o = {}) => {
    const th = Object.assign({ bg: "transparent", edge: "#9aa6b2", text: "#1d232b", sub: "#5b6672", alarm: "#d03b3b", font: "inherit",
      states: LAB.STATES, radius: 8 }, o.theme || {});
    svg.innerHTML = "";
    if (!nodes || !nodes.length) { svg.setAttribute("viewBox", "0 0 400 60"); el("text", { x: 12, y: 34, "font-size": 13, fill: th.sub }, svg, "No plan reported yet."); return { W: 400, H: 60, pos: {} }; }
    const L = LAB.layers(nodes), cols = Math.max(...Object.values(L)) + 1;
    const byCol = {}; nodes.forEach((n) => (byCol[L[n.id]] = byCol[L[n.id]] || []).push(n));
    const colW = o.colW || 214, nw = o.nodeW || 184, nh = o.nodeH || 58, gap = o.gap || 20;
    const W = Math.max(o.minW || 600, 24 + cols * colW), H = Math.max(...Object.values(byCol).map((c) => c.length)) * (nh + gap) + 30;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); if (!o.fit) { svg.setAttribute("width", W); svg.setAttribute("height", H); }
    if (th.bg !== "transparent") el("rect", { x: 0, y: 0, width: W, height: H, fill: th.bg }, svg);
    const pos = {};
    Object.entries(byCol).forEach(([c, list]) => { const tot = list.length * (nh + gap) - gap, y0 = (H - tot) / 2;
      list.forEach((n, i) => (pos[n.id] = { x: 12 + c * colW, y: y0 + i * (nh + gap) })); });
    const id = "arr" + Math.random().toString(36).slice(2, 7);
    const mk = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, el("defs", {}, svg));
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: th.edge }, mk);
    nodes.forEach((n) => (n.needs || []).forEach((s) => { if (!pos[s]) return; const a = pos[s], b = pos[n.id];
      const x1 = a.x + nw, y1 = a.y + nh / 2, x2 = b.x - 2, y2 = b.y + nh / 2, mx = (x1 + x2) / 2;
      el("path", { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, fill: "none", stroke: th.edge, "stroke-width": 1.5,
        opacity: n.state === "dropped" ? 0.4 : 1, "marker-end": `url(#${id})` }, svg); }));
    nodes.forEach((n) => {
      const p = pos[n.id], st = th.states[n.state], g = el("g", { style: "cursor:pointer" }, svg);
      const stroke = n.ev === "bad" ? th.alarm : st.stroke, dash = n.ev === "rep" ? "5 4" : "";
      el("rect", { x: p.x, y: p.y, width: nw, height: nh, rx: th.radius, fill: st.fill, stroke, "stroke-width": n.ev === "bad" || n.id === o.selected ? 2.6 : 1.6, "stroke-dasharray": dash }, g);
      const words = n.title.split(" "), lines = [""];
      words.forEach((w) => { if ((lines[lines.length - 1] + " " + w).trim().length > (o.wrap || 25) && lines.length < 2) lines.push(w); else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim(); });
      lines.forEach((ln, i) => { const t = el("text", { x: p.x + 9, y: p.y + 17 + i * 14, "font-size": 12, fill: n.state === "dropped" ? st.ink : th.text, "font-weight": 600, "font-family": th.font }, g, ln.length > 28 ? ln.slice(0, 27) + "…" : ln);
        if (n.state === "dropped") t.setAttribute("text-decoration", "line-through"); });
      el("text", { x: p.x + 9, y: p.y + nh - 9, "font-size": 10.5, fill: n.ev === "bad" ? th.alarm : th.sub, "font-family": th.font }, g, `${st.label} · ${n.kind} · ${LAB.EVIDENCE[n.ev].short}`);
      if (o.prevIds && !o.prevIds.has(n.id)) { el("rect", { x: p.x + nw - 36, y: p.y - 8, width: 34, height: 15, rx: 7, fill: th.newBadge || "#2a78d6" }, g); el("text", { x: p.x + nw - 19, y: p.y + 3, "font-size": 9.5, fill: "#fff", "text-anchor": "middle" }, g, "NEW"); }
      g.addEventListener("click", () => o.onClick && o.onClick(n));
      el("title", {}, g, `${n.title}\n${st.label} — ${LAB.EVIDENCE[n.ev].label}\n${n.evidence || ""}`);
    });
    return { W, H, pos };
  };

  // --- the proposal: a real document ---------------------------------------------------------------------------------
  D.proposal = {
    title: "Neural post-processing of station temperature forecasts",
    subtitle: "Research proposal · study stations · drafted by lead-7c0f · approved (signed) 2026-10-12 08:02 · amendment 1",
    blocks: [
      ["h2", "1. Question"],
      ["p", "Do neural post-processing models beat ensemble model output statistics (EMOS) for 24-hour 2 m temperature forecasts at 312 weather stations, over 2019–2024? EMOS is the operational standard; recent work claims distributional regression networks (DRN) reduce the mean absolute error by 8–12%."],
      ["h2", "2. Hypotheses and what would falsify them"],
      ["li", "H1. A DRN with station embeddings beats EMOS on MAE. Falsified if the paired bootstrap test against EMOS does not reject at the sealed alpha (alpha-investing, wealth 0.25)."],
      ["li", "H2. A quantile regression forest beats EMOS on CRPS. Falsified the same way, on CRPS."],
      ["li", "H3. Lagged observations improve either model. Falsified if the comparison's interval contains zero; the truncation audit must be clean."],
      ["h2", "3. Data"],
      ["p", "The station panel comes from the provider snapshot of 2026-09-30, with known_at set to each forecast's issue time. Research covers 2019–2023; the holdout is 2024, with a one-day embargo. No feature may use an observation that was not known when the forecast was issued."],
      ["h2", "4. Branches and budgets"],
      ["table", [["Branch", "Method", "Budget", "Trials", "Status"], ["A", "DRN with station embeddings", "$2 · 60 min", "20", "running"], ["B", "Quantile regression forest", "$2 · 60 min", "20", "queued for 2 GPUs"], ["C", "Analog ensemble", "$1 · 30 min", "10", "dropped (dead end)"]]],
      ["h2", "5. Selection and the holdout look"],
      ["p", "The final candidate is chosen by the sealed rule seed_robust with k = 3: the configuration with the best mean MAE over at least three seeds. A person then takes one holdout look on that run. No other look is planned."],
      ["h2", "6. Stop and kill rules"],
      ["p", "A branch stops after six runs without a new best. A branch is killed if its paired test favours EMOS. The study stops when the trial budget of 40 is spent or the clock reaches the consolidate stage."],
      ["h2", "7. Risks"],
      ["p", "Station metadata changes (relocations) may break the panel's comparability across years. The DRN may overfit stations with short histories; the seed-robust rule and the holdout look are the guard."],
    ],
  };
  LAB.renderDoc = (docEl) => {
    const P = D.proposal; let h = `<h1 class="lab-doc-title">${P.title}</h1><p class="lab-doc-sub">${P.subtitle}</p>`;
    let inList = false;
    P.blocks.forEach(([k, v], i) => {
      if (k !== "li" && inList) { h += "</ul>"; inList = false; }
      if (k === "h2") h += `<h2 data-b="${i}">${v}</h2>`;
      else if (k === "p") h += `<p data-b="${i}">${v}</p>`;
      else if (k === "li") { if (!inList) { h += "<ul>"; inList = true; } h += `<li data-b="${i}">${v}</li>`; }
      else if (k === "table") h += `<table data-b="${i}">${v.map((r, j) => `<tr>${r.map((c) => j ? `<td>${c}</td>` : `<th>${c}</th>`).join("")}</tr>`).join("")}</table>`;
    });
    if (inList) h += "</ul>";
    docEl.innerHTML = h;
  };

  // --- Word-style review: highlights in the text, comments in the margin ---------------------------------------------
  D.docComments = [
    { id: "c1", quote: "reduce the mean absolute error by 8–12%", who: "sam", when: "08:55", text: "Cite the papers behind 8–12%, with their test sets: is it the same kind of station data?",
      replies: [{ who: "lead-7c0f", agent: true, when: "08:58", text: "Added: Rasp & Lerch 2018 (Germany, 537 stations) and two follow-ups; quoted on their pages in the review." }], delivered: true },
    { id: "c2", quote: "the truncation audit must be clean", who: "sam", when: "09:03", text: "Good. Run 3c19 already failed this audit: say in the plan how the fix is made (op debug), not just that it is.", replies: [], delivered: true },
    { id: "c3", quote: "seed_robust with k = 3", who: "sam", when: "09:12", text: "Three seeds is the minimum. Can the DRN afford five within budget?",
      replies: [{ who: "lead-7c0f", agent: true, when: "09:13", text: "Five seeds × 9 min ≈ 45 min: within branch A's 60. exp-a1 will take seeds 4 and 5 once 2 and 3 finish." }], delivered: true },
  ];
  function textNodesIn(range) {
    const out = [], walker = document.createTreeWalker(range.commonAncestorContainer.nodeType === 3 ? range.commonAncestorContainer.parentNode : range.commonAncestorContainer, NodeFilter.SHOW_TEXT);
    let n; while ((n = walker.nextNode())) if (range.intersectsNode(n) && n.textContent.trim()) out.push(n);
    return out;
  }
  function wrapRange(range, id) {
    const marks = [];
    textNodesIn(range).forEach((t) => {
      let s = 0, e = t.textContent.length;
      if (t === range.startContainer) s = range.startOffset;
      if (t === range.endContainer) e = range.endOffset;
      if (e <= s) return;
      const r = document.createRange(); r.setStart(t, s); r.setEnd(t, e);
      const m = document.createElement("mark"); m.className = "lab-hl"; m.dataset.c = id; r.surroundContents(m); marks.push(m);
    });
    return marks;
  }
  function findQuote(docEl, quote) {
    const walker = document.createTreeWalker(docEl, NodeFilter.SHOW_TEXT); let n;
    while ((n = walker.nextNode())) { const i = n.textContent.indexOf(quote); if (i >= 0) { const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + quote.length); return r; } }
    return null;
  }
  LAB.annotate = (docEl, marginEl, opts = {}) => {
    if (docEl._labAnnotator) return docEl._labAnnotator;          // once per document: one button, one set of listeners
    const author = opts.author || "you", agent = opts.agent || "lead-7c0f";
    const comments = D.docComments;
    let active = null;
    const btn = document.createElement("button"); btn.className = "lab-commentbtn"; btn.textContent = opts.button || "Comment"; btn.style.position = "absolute"; btn.style.display = "none";
    document.body.appendChild(btn);
    const layout = () => {                                       // a no-op while hidden; call it again once shown
      if (!docEl.offsetParent && getComputedStyle(docEl).position !== "fixed") return;
      const dr = docEl.getBoundingClientRect();
      const scale = docEl.offsetHeight ? dr.height / docEl.offsetHeight || 1 : 1;   // inside a CSS scale() (a zoomed canvas)
      const base = marginEl.getBoundingClientRect().top; let floor = 0;
      [...marginEl.querySelectorAll(".lab-card")].map((card) => {
        const m = docEl.querySelector(`mark[data-c="${card.dataset.c}"]`);
        const want = m ? (m.getBoundingClientRect().top - base) / scale : floor;
        return { card, want };
      }).sort((a, b) => a.want - b.want).forEach(({ card, want }) => { const top = Math.max(want, floor); card.style.top = top + "px"; floor = top + card.offsetHeight + 8; });
      marginEl.style.minHeight = floor + "px";
    };
    const setActive = (id) => { active = id; docEl.querySelectorAll("mark.lab-hl").forEach((m) => m.classList.toggle("lab-on", m.dataset.c === id));
      marginEl.querySelectorAll(".lab-card").forEach((c) => c.classList.toggle("lab-on", c.dataset.c === id)); layout(); };
    const card = (c) => {
      const d = document.createElement("div"); d.className = "lab-card" + (c.resolved ? " lab-resolved" : ""); d.dataset.c = c.id; d.style.position = "absolute"; d.style.left = "0"; d.style.right = "0";
      const msg = (m) => `<div class="lab-msg"><span class="lab-who${m.agent ? " lab-agent" : ""}">${esc(m.who)}</span><span class="lab-when">${esc(m.when)}</span><div class="lab-text">${esc(m.text)}</div>${m.delivered ? `<div class="lab-deliv">✓ delivered to ${esc(agent)} as advice (tell → told)</div>` : ""}</div>`;
      d.innerHTML = c.draft
        ? `<div class="lab-quote">“${esc(c.quote.length > 70 ? c.quote.slice(0, 69) + "…" : c.quote)}”</div><textarea class="lab-input" placeholder="Comment for ${esc(agent)}…"></textarea>
           <div class="lab-actions"><button class="lab-cancel">Cancel</button><button class="lab-post">Comment</button></div>`
        : `${msg(c)}${c.replies.map(msg).join("")}
           ${c.resolved ? `<div class="lab-deliv">resolved</div>` : `<div class="lab-reply"><input class="lab-input" placeholder="Reply…"><button class="lab-replybtn">Reply</button><button class="lab-resolve">Resolve</button></div>`}`;
      d.addEventListener("click", () => setActive(c.id));
      if (c.draft) {
        d.querySelector(".lab-cancel").onclick = (e) => { e.stopPropagation(); docEl.querySelectorAll(`mark[data-c="${c.id}"]`).forEach((m) => m.replaceWith(...m.childNodes)); comments.splice(comments.indexOf(c), 1); render(); };
        d.querySelector(".lab-post").onclick = (e) => { e.stopPropagation(); const v = d.querySelector("textarea").value.trim(); if (!v) return;
          Object.assign(c, { draft: false, text: v, who: author, when: new Date().toTimeString().slice(0, 5), delivered: true });
          LAB.emit("feed", { who: author, text: `comment on “${c.quote.slice(0, 40)}…” → ${agent} (advice)` }); render(); setActive(c.id); };
        setTimeout(() => d.querySelector("textarea").focus(), 0);
      } else if (!c.resolved) {
        const inp = d.querySelector(".lab-reply input");
        const reply = () => { const v = inp.value.trim(); if (!v) return;
          c.replies.push({ who: author, when: new Date().toTimeString().slice(0, 5), text: v, delivered: true });
          LAB.emit("feed", { who: author, text: `reply on “${c.quote.slice(0, 40)}…” → ${agent} (advice)` }); render(); setActive(c.id); };
        inp.addEventListener("keydown", (e) => { if (e.key === "Enter") reply(); });
        d.querySelector(".lab-replybtn").onclick = (e) => { e.stopPropagation(); reply(); };
        d.querySelector(".lab-resolve").onclick = (e) => { e.stopPropagation(); c.resolved = true; docEl.querySelectorAll(`mark[data-c="${c.id}"]`).forEach((m) => m.classList.add("lab-res")); render(); };
      }
      return d;
    };
    const render = () => { marginEl.innerHTML = ""; comments.forEach((c) => marginEl.appendChild(card(c))); layout(); opts.onChange && opts.onChange(comments); LAB.emit("comments", comments); };
    comments.forEach((c) => { if (!docEl.querySelector(`mark[data-c="${c.id}"]`)) { const r = findQuote(docEl, c.quote); if (r) wrapRange(r, c.id).forEach((m) => c.resolved && m.classList.add("lab-res")); } });
    docEl.addEventListener("click", (e) => { const m = e.target.closest("mark.lab-hl"); if (m) setActive(m.dataset.c); });
    document.addEventListener("mouseup", (e) => {
      if (e.target === btn) return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !docEl.contains(sel.anchorNode) || !docEl.contains(sel.focusNode) || !sel.toString().trim()) { btn.style.display = "none"; return; }
      const r = sel.getRangeAt(0).getBoundingClientRect();
      btn.style.left = r.right + window.scrollX + 6 + "px"; btn.style.top = r.top + window.scrollY - 30 + "px"; btn.style.display = "block";
    });
    btn.addEventListener("mousedown", (e) => e.preventDefault());
    btn.addEventListener("click", () => {
      const sel = window.getSelection(); if (!sel || sel.isCollapsed) return;
      const id = "c" + Math.random().toString(36).slice(2, 7), quote = sel.toString().trim();
      wrapRange(sel.getRangeAt(0), id); sel.removeAllRanges(); btn.style.display = "none";
      comments.push({ id, quote, draft: true, replies: [] }); render(); setActive(id);
    });
    const follow = () => {                                       // keep the button by the selection when a pane scrolls
      if (btn.style.display === "none") return;
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !docEl.contains(sel.anchorNode)) { btn.style.display = "none"; return; }
      const r = sel.getRangeAt(0).getBoundingClientRect();
      btn.style.left = r.right + window.scrollX + 6 + "px"; btn.style.top = r.top + window.scrollY - 30 + "px";
    };
    document.addEventListener("scroll", follow, true);
    window.addEventListener("resize", layout);
    render();
    return (docEl._labAnnotator = { layout, setActive, button: btn });
  };

  // --- the rest: feed, inbox, queue, spawn ---------------------------------------------------------------------------
  D.feed = [
    ["09:14:02", "exp-a1", "study_run stations --op improve --parent run:4fd2 --seed 2 (queue ticket 41)"],
    ["09:13:40", "lead-7c0f", "agent_plan v3: + quantile-forest branch; the selection needs t1, t2 and the critique"],
    ["09:12:40", "lead-7c0f", "told: “Three seeds is the minimum…” (a comment by sam on the proposal)"],
    ["09:10:55", "exp-b2", "queued: waiting for gpus held by running runs or reserved"],
    ["09:07:31", "rev-c3", "study_critique on run:4fd2: 3 findings"],
    ["09:02:10", "lead-7c0f", "inbox #7 holdout: after the selection, one look on run:4fd2"],
  ];
  LAB.on("feed", (f) => D.feed.unshift([new Date().toTimeString().slice(0, 8), f.who, f.text]));
  // The inbox: what waits for the person. `approval` items are a person's act (approve or deny, signed, with an optional
  // message to the agent); `choice` items are an agent's question, single (radio) or multiple (checkbox), where `other`
  // lets the person type an answer of their own.
  D.inbox = [
    { id: 7, type: "approval", kind: "holdout", from: "lead-7c0f", at: "09:02", person: true, step: "look",
      what: "After the selection: one holdout look on run:4fd2", detail: "The look spends 1 of 1 allowed; it scores the 2024 holdout once." },
    { id: 8, type: "choice", kind: "question", from: "exp-a1", at: "09:15", step: "d3", multi: false, other: true,
      what: "How many seeds for the final DRN candidate?", detail: "The sealed rule needs at least 3. Each seed takes about 9 minutes; branch A has 41 minutes left.",
      choices: ["3 seeds — the sealed minimum", "5 seeds — about 45 minutes, within the branch budget", "10 seeds — over the branch budget; needs a budget request"] },
    { id: 9, type: "choice", kind: "question", from: "lead-7c0f", at: "09:16", step: "paper", multi: true, other: true,
      what: "Which baselines should the write-up report beside EMOS?", detail: "Each is a recorded run already; reporting costs no trial.",
      choices: ["Persistence", "Climatology", "Raw ensemble mean", "Analog ensemble (the dropped branch)"] },
    { id: 10, type: "approval", kind: "budget", from: "exp-b2", at: "09:17", person: true, step: "q1",
      what: "Raise branch B's budget from $2.00 to $3.00", detail: "The GPU queue cost 4 minutes; QRF needs about 55 minutes on 2 GPUs." },
  ];
  D.answered = [];
  LAB.answer = (id, { decision, message = "", picks = [], other = "" } = {}) => {
    const i = D.inbox.findIndex((x) => x.id === id); if (i < 0) return null;
    const it = D.inbox.splice(i, 1)[0];
    const a = { ...it, decision, message: message.trim(), picks, other: other.trim(), answeredAt: new Date().toTimeString().slice(0, 5), by: "you" };
    D.answered.unshift(a);
    const what = it.type === "choice" ? `answered #${id}: ${[...picks, ...(a.other ? [`other: “${a.other}”`] : [])].join("; ") || "(nothing)"}`
      : `${decision === "approve" ? "approved" : "denied"} #${id} (${it.kind}, signed)`;
    LAB.emit("feed", { who: "you", text: what + (a.message ? ` — “${a.message}”` : "") + ` → ${it.from}` });
    LAB.emit("inbox", D.inbox);
    return a;
  };
  D.queue = [
    { n: 41, state: "running", for: "18.2m", needs: "4c 24GB 1gpu ~40m", run: "stations research", agent: "exp-a1", why: "" },
    { n: 42, state: "running", for: "6.0m", needs: "1c 0GB 0gpu ~3m", run: "stations research", agent: "lead-7c0f", why: "" },
    { n: 43, state: "waiting", for: "4.1m", needs: "4c 24GB 2gpu ~55m", run: "stations research", agent: "exp-b2", why: "waiting for gpus held by running runs or reserved" },
  ];
  D.settings = [
    ["default budget", "$2.00 · 60 min · 200 turns", "openlab.toml [agents]"],
    ["caps", "$20 · 480 min · depth 3 · 6 children", "openlab.toml [agents]"],
    ["notify", "notify-send openlab", "openlab.toml [agents]"],
    ["deny patterns", "kaggle … download, prepared/private", "openlab.toml [agents]"],
    ["W&B mirror", "on · project openlab", "openlab.toml [mirror]"],
    ["box availability", "until 2026-10-20 08:00 · GPU 1 reserved 14:00–18:00", "~/.openlab/box/box.toml"],
  ];
  LAB.spawn = ({ role = "experimenter", prompt = "", usd = 2, minutes = 60, model = "", parent = "lead-7c0f" } = {}) => {
    const top = !parent || parent === "(none)";
    const s = { id: `${role.slice(0, 3)}-${Math.random().toString(16).slice(2, 6)}`, role, state: "starting", usd: 0, budget: +usd || 1,
      minutes: +minutes || 60, model: model || "default", parent: top ? null : parent, study: "stations",
      q: prompt.length > 90 ? prompt.slice(0, 89) + "…" : prompt || "(no prompt)", last: "starting", ago: "now" };
    D.sessions.push(s); D.plans[s.id] = [];
    LAB.emit("feed", { who: "you", text: `spawned ${s.id} (${role}, $${s.budget}, ${s.minutes} min${model ? ", " + model : ""})${top ? "" : " under " + parent}` });
    LAB.emit("sessions", D.sessions);
    return s;
  };
  LAB.declare = ({ id, role = "", cap = 0, key = false, step = "" } = {}) => {
    const p = { id, kind: "person", role: role || "contributor", state: "available", hours: 0, cap: +cap || 0, key: !!key,
      q: step || "(nothing assigned yet)", last: "declared", ago: "now",
      plans: step ? [{ v: 1, note: "declared by you", nodes: [N("p1", step, "review", [], "planned", "rep", "")] }] : [] };
    D.people.push(p);
    LAB.emit("feed", { who: "you", text: `declared ${id} (${p.role}${p.cap ? `, ${p.cap} h a week` : ""}${key ? ", holds a person key" : ""})` });
    LAB.emit("sessions", D.sessions);
    return p;
  };
  // the review styles every template restyles: these are the defaults
  const css = document.createElement("style");
  css.textContent = `mark.lab-hl{background:rgba(255,213,79,.45);color:inherit;cursor:pointer;border-radius:2px}
  mark.lab-hl.lab-on{background:rgba(255,183,0,.75)} mark.lab-hl.lab-res{background:rgba(160,160,160,.18)}
  .lab-commentbtn{z-index:10000;font:12px sans-serif;padding:3px 9px;border-radius:5px;border:1px solid #c9a227;background:#fff7d6;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,.15)}
  .lab-card{background:#fff;border:1px solid #d8dee6;border-radius:6px;padding:7px 9px;font-size:12.5px;box-shadow:0 1px 3px rgba(0,0,0,.06);transition:top .15s}
  .lab-card.lab-on{border-color:#c9a227;box-shadow:0 2px 10px rgba(201,162,39,.25)} .lab-card.lab-resolved{opacity:.55}
  .lab-who{font-weight:600}.lab-agent{color:#1c5cab}.lab-when{opacity:.6;margin-left:6px;font-size:11px}.lab-msg{padding:3px 0}
  .lab-msg+.lab-msg{border-top:1px solid #eef1f4}.lab-deliv{font-size:11px;color:#0a6e0a;margin-top:3px}.lab-quote{font-style:italic;opacity:.7;margin-bottom:4px}
  .lab-input{width:100%;box-sizing:border-box;font-family:inherit;font-size:12.5px;border:1px solid #c9d3e0;border-radius:4px;padding:4px}
  textarea.lab-input{min-height:48px}.lab-actions,.lab-reply{display:flex;gap:6px;justify-content:flex-end;margin-top:4px}
  .lab-actions button,.lab-reply button{font-family:inherit;font-size:12px;padding:2px 8px;border-radius:4px;border:1px solid #b9c5d6;background:#fff;cursor:pointer}
  .lab-card .lab-actions .lab-post{background:#2a78d6;color:#fff;border-color:#2a78d6}`;
  document.head.insertBefore(css, document.head.firstChild);   // first, so a template's own rules win at equal specificity
})();
