/* openlab show, live: when the page is served by `openlab show` (a ?t= token in its URL), the demo data in core.js
   is replaced by the workspace's own, read from the server's show_snapshot tool, and read again whenever the server
   says the control store changed (/ws/events). Opened as a file, the page stays the demo.

   LAB.live            {token, reviewer, ws} when live; ws is the research (workspace) the page shows
   LAB.api(name, args) one tool call in that research: the dispatch envelope {ok, result | error, meaning, try}
   LAB.wsList          the researches under the server's home, for the tabs; LAB.switchWs(name) opens one
   LAB.newResearch(b)  a new research: its folder, openlab init, and its lead agent
   LAB.fileUrl(path)   a link to a file of the research (a paper's PDF, the code folder)
   LAB.load()          read the snapshot again, then LAB.emit("live")
   LAB.now()           the server's clock, in ms */
(function () {
  const q = new URLSearchParams(location.search), token = q.get("t");
  if (!token || location.protocol === "file:") { LAB.now = () => Date.now(); return; }
  const reviewer = q.get("r") || "", D = LAB.data;
  LAB.live = { token, reviewer, ws: q.get("w") || "" };
  const empty = () => ({ live: true, sessions: [], people: [], plans: {}, feed: [], inbox: [], answered: [], queue: [],
    settings: [], docComments: [], agst: {}, ledger: [], ledgerHeads: [], spend: 0 });
  Object.assign(D, empty());
  let skew = 0;
  LAB.now = () => Date.now() + skew;

  LAB.api = async (name, args = {}) => {
    try {
      const res = await fetch(`/api/tool/${name}?ws=${encodeURIComponent(LAB.live.ws)}`, { method: "POST", body: JSON.stringify(args),
        headers: { Authorization: "Bearer " + token, "Content-Type": "application/json", "x-openlab-reviewer": reviewer } });
      const env = await res.json();
      if (env && env.error === "a reviewer's act") { LAB.stale = true; LAB.emit("stale-link", true); }
      return env;
    } catch (e) { return { ok: false, error: "offline", meaning: "the openlab server did not answer", try: "is openlab show still running?" }; }
  };
  LAB.checkLink = async () => {                       // a link from an earlier run of openlab show may read, not act
    try {
      const r = await (await fetch("/api/person", { headers: { Authorization: "Bearer " + token, "x-openlab-reviewer": reviewer } })).json();
      LAB.stale = !r.person; LAB.emit("stale-link", LAB.stale);
    } catch (e) { /* offline: the next connect checks again */ }
  };

  const auth = { Authorization: "Bearer " + token, "Content-Type": "application/json" };
  LAB.workspaces = async () => {
    try {
      const r = await (await fetch("/api/workspaces", { headers: auth })).json();
      LAB.wsList = r.workspaces; LAB.home = r.home.replace(/^\/home\/[^/]+/, "~"); LAB.defaults = r.settings || [];
      const open = r.workspaces.filter((w) => !w.closed);             // a closed tab is hidden, its research untouched
      if (!LAB.live.ws || !open.some((w) => w.name === LAB.live.ws)) { LAB.live.ws = r.default || (open[0] || {}).name || ""; Object.assign(D, empty()); }
      LAB.emit("workspaces", LAB.wsList);
    } catch (e) { /* the next event tries again */ }
  };
  const remember = () => { const u = new URLSearchParams(location.search); u.set("w", LAB.live.ws); history.replaceState(null, "", `${location.pathname}?${u}${location.hash}`); };
  LAB.switchWs = (name) => { LAB.live.ws = name; remember(); Object.assign(D, empty()); LAB.emit("workspaces", LAB.wsList); return LAB.load(); };
  LAB.newResearch = async (body) => {
    try { const r = await fetch("/api/workspaces", { method: "POST", headers: auth, body: JSON.stringify(body) }); const env = await r.json();
      if (env.ok) await LAB.workspaces(); return env; }
    catch (e) { return { ok: false, error: "offline", meaning: "the openlab server did not answer" }; }
  };
  LAB.startLead = async (name, body) => {             // a research whose lead did not start, or has ended: a new lead
    try { const r = await fetch(`/api/workspaces/${encodeURIComponent(name)}/lead`, { method: "POST", headers: auth, body: JSON.stringify(body) });
      return await r.json(); }
    catch (e) { return { ok: false, error: "offline", meaning: "the openlab server did not answer" }; }
  };
  const person = { ...auth, "x-openlab-reviewer": reviewer };
  const call = async (url, opts) => {
    try { return await (await fetch(url, opts)).json(); }
    catch (e) { return { ok: false, error: "offline", meaning: "the openlab server did not answer" }; }
  };
  LAB.setTab = async (name, open) => {                // close a research's tab, or open it again (Resume)
    const env = await call(`/api/workspaces/${encodeURIComponent(name)}/tab`, { method: "POST", headers: auth, body: JSON.stringify({ open }) });
    if (env.ok) await LAB.workspaces(); return env;
  };
  LAB.machines = () => call("/api/machines", { headers: auth });
  LAB.editMachines = (body) => call("/api/machines", { method: "POST", headers: person, body: JSON.stringify(body) });
  LAB.fileUrl = (path) => `/api/file?ws=${encodeURIComponent(LAB.live.ws)}&path=${encodeURIComponent(path)}&t=${encodeURIComponent(token)}`;
  const sec = (iso) => (iso ? Math.floor(Date.parse(iso) / 1000) : 0);
  const pad = (n) => String(n).padStart(2, "0");
  const today = (d) => d.toDateString() === new Date(LAB.now()).toDateString();
  LAB.hms = (iso) => { if (!iso) return ""; const d = new Date(iso); return `${today(d) ? "" : `${d.getMonth() + 1}-${pad(d.getDate())} `}${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
  LAB.hm = (iso) => LAB.hms(iso).slice(0, -3);

  function adapt(s) {
    skew = Date.parse(s.now) - Date.now();
    D.workspace = s.workspace.replace(/^\/home\/[^/]+/, "~");
    D.box = { host: s.box.host, gpus: s.box.gpus, until: s.box.until || "", running: s.box.running, waiting: s.box.waiting,
      reservations: s.box.reservations || [] };
    D.spend = s.spend;
    D.sessions = s.sessions.map((x) => ({ id: x.id, role: x.role, state: x.state, usd: x.usd, budget: x.budget || 0, parent: x.parent,
      study: x.study, q: x.q || x.now.op, last: x.now.op, ago: "", model: x.model, flags: x.flags, trials: x.trials, claims: x.claims,
      usd_estimated: x.usd_estimated, lead: !!x.lead, held: x.held || "" }));
    D.agst = {};
    s.sessions.forEach((x) => (D.agst[x.id] = { model: x.model || x.cli || "", started: sec(x.started), turns: x.turns, tools: x.tools,
      refusals: x.refusals, refused: x.refused || 0, bypass: x.bypass || 0, attempts: (x.attempts || []).map((r) => ({ ...r, t: LAB.hms(r.at) })),
      flags: x.flags, now: { kind: x.now.kind, op: x.now.op, note: x.now.note || "", since: sec(x.now.since) }, heard: sec(x.heard) }));
    D.people = s.people.map((p) => ({ ...p, plans: [] }));
    D.people.forEach((p) => (D.agst[p.id] = { now: { kind: p.state === "away" ? "away" : "person", op: p.state === "away" ? `away until ${p.until}` : p.q || "available",
      note: "", since: 0 }, heardText: "—" }));
    D.plans = {};
    const ended = {};                                // an ended agent's steps stop at its end on the chart, not at now
    s.sessions.forEach((x) => { if (["done", "failed", "stopped", "lost", "ended"].includes(x.state)) ended[x.id] = x.now.since; });
    Object.entries(s.plans).forEach(([id, vs]) => (D.plans[id] = vs.map((v) => ({ v: v.v, note: v.note, at: v.at,
      nodes: v.nodes.map((n) => ({ ...n, needs: n.needs || [], cap: ended[id] || null })) }))));
    D.feed = s.feed.map((e) => [LAB.hms(e.at), e.who, e.text]);
    D.inbox = s.inbox.map((i) => ({ ...i, at: LAB.hm(i.at) }));
    D.answered = s.answered.map((a) => ({ ...a, answeredAt: LAB.hm(a.answeredAt) }));
    D.queue = s.queue;
    D.settings = s.settings;
    D.ledger = s.ledger.entries.map((e, i) => ({ seq: i + 1, rseq: e.seq, study: e.study, day: "", t: LAB.hms(e.at), at: e.at, who: e.who,
      kind: e.kind, id: e.id, text: e.text, sig: e.sig, bad: e.bad, tests: e.tests || [], breaks: e.breaks || [], needed_by: e.needed_by || 0, prev: e.prev.slice(0, 8), hash: e.hash.slice(0, 8) }));
    D.ledgerHeads = s.ledger.heads;
    D.ledgerTotal = s.ledger.total;
  }

  let loading = null, again = false;
  LAB.load = async () => {
    if (loading) { again = true; return loading; }
    loading = (async () => {
      if (!LAB.wsList) { await LAB.workspaces(); remember(); }
      if (!LAB.live.ws) { LAB.emit("live-none"); return; }            // every tab is closed: New session opens one
      const env = await LAB.api("show_snapshot");
      if (env.ok) { adapt(env.result); LAB.emit("live", D); }
      else LAB.emit("live-error", env);
    })();
    await loading; loading = null;
    if (again) { again = false; return LAB.load(); }
  };

  // the server publishes a `control` event when an agent writes the control store, and `workspace` after a tool call
  let timer = null, tabs = null;
  const soon = () => { clearTimeout(timer); timer = setTimeout(LAB.load, 250); };
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/events?t=${encodeURIComponent(token)}`);
    ws.onmessage = (m) => {
      let e; try { e = JSON.parse(m.data); } catch (_) { return; }
      if (e.channel === "hello") return;
      if (e.channel === "workspaces") return LAB.workspaces();
      if (!e.ws || e.ws === LAB.live.ws) soon();                    // this research changed
      clearTimeout(tabs); tabs = setTimeout(LAB.workspaces, 1500);  // and the tabs' counts
    };
    ws.onopen = () => { LAB.emit("live-link", true); LAB.checkLink(); };          // a reconnect may be a new run
    ws.onclose = () => { LAB.emit("live-link", false); setTimeout(connect, 2000); };
  }
  connect();
  setInterval(LAB.load, 30000);                      // and once in a while regardless, for the clocks
})();
