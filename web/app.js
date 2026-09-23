const $ = (s) => document.querySelector(s);
const messages = $("#messages");
const input = $("#input");
const conn = $("#conn");

function addMsg(role, text, meta) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  if (meta) {
    const m = document.createElement("span");
    m.className = "meta";
    m.textContent = meta;
    el.appendChild(m);
  }
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || r.statusText);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

async function refreshHealth() {
  try {
    const h = await api("/health");
    conn.textContent = "online";
    conn.className = "pill online";
    $("#stHealth").textContent = "OK";
    $("#stBackend").textContent = h.backend || h.status || "ok";
    $("#version").textContent = h.version ? `v${h.version}` : "mmap · low RAM";
    if (h.model) $("#modelPath").placeholder = h.model;
  } catch {
    conn.textContent = "offline";
    conn.className = "pill offline";
    $("#stHealth").textContent = "down";
  }
}

async function listModels() {
  const box = $("#modelList");
  box.innerHTML = "Scanning…";
  try {
    const data = await api("/v1/models");
    box.innerHTML = "";
    (data.models || []).forEach((m) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${m.size_mb} MB · ${m.name}`;
      b.title = m.path;
      b.onclick = () => { $("#modelPath").value = m.path; };
      box.appendChild(b);
    });
    if (!(data.models || []).length) box.textContent = "No GGUF found";
  } catch (e) {
    box.textContent = String(e.message || e);
  }
}

$("#btnListModels").onclick = listModels;
$("#btnNew").onclick = async () => {
  messages.innerHTML = "";
  try { await api("/v1/reset", { method: "POST", body: "{}" }); } catch {}
  addMsg("sys", "New chat");
};
$("#btnDoctor").onclick = async () => {
  const typing = addMsg("bot", "Running doctor…", "");
  typing.classList.add("typing");
  try {
    const d = await api("/v1/doctor");
    typing.classList.remove("typing");
    typing.textContent = d.text || JSON.stringify(d, null, 2);
  } catch (e) {
    typing.textContent = String(e.message || e);
  }
};
$("#menuBtn").onclick = () => $("#sidebar").classList.toggle("open");

$("#composer").onsubmit = async (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  autoSize();
  addMsg("user", text);
  const typing = addMsg("bot", "…", "");
  typing.classList.add("typing");
  const body = {
    message: text,
    agent: $("#agentMode").checked,
    extreme_low_ram: $("#extreme").checked,
    ram_budget_mb: Number($("#ramBudget").value) || 0,
    model: $("#modelPath").value.trim() || undefined,
  };
  try {
    const data = await api("/v1/chat", { method: "POST", body: JSON.stringify(body) });
    typing.classList.remove("typing");
    typing.textContent = data.reply || data.text || "(empty)";
    const meta = [];
    if (data.elapsed_s != null) meta.push(`${Number(data.elapsed_s).toFixed(1)}s`);
    if (data.rss_mb != null) meta.push(`RSS ${Number(data.rss_mb).toFixed(0)} MB`);
    if (data.backend) meta.push(data.backend);
    if (meta.length) {
      const m = document.createElement("span");
      m.className = "meta";
      m.textContent = meta.join(" · ");
      typing.appendChild(m);
    }
    if (data.rss_mb != null) $("#stRss").textContent = `${Number(data.rss_mb).toFixed(0)} MB`;
    if (data.tool_trace && data.tool_trace.length) {
      addMsg("sys", "tools: " + data.tool_trace.map((t) => t.name).join(", "));
    }
  } catch (e) {
    typing.classList.remove("typing");
    typing.textContent = "Error: " + (e.message || e);
  }
};

function autoSize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
input.addEventListener("input", autoSize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#composer").requestSubmit();
  }
});

addMsg("sys", "DiskChat ready — pick a GGUF and say hello.");
refreshHealth();
setInterval(refreshHealth, 15000);
