// Long-running tasks panel: upload a big file, watch chunk progress, pause / resume / cancel, download DOCX/PDF/TXT/MD.
(() => {
  const overlay = document.getElementById("tasksOverlay");
  if (!overlay) return;
  const list = document.getElementById("tasksList");
  const statusLine = document.getElementById("taskStatus");
  const fileInput = document.getElementById("taskFile");
  const startButton = document.getElementById("taskStart");
  let timer = null;

  const STATE_LABEL = {
    queued: "در صف", preparing: "آماده‌سازی", running: "در حال اجرا", paused: "متوقف‌شده", cancelled: "لغوشده",
    completed: "تمام‌شده", partial: "ناقص (بخش‌هایی ناموفق)", failed: "ناموفق", interrupted: "قطع‌شده؛ قابل ادامه",
  };
  const ACTIVE = new Set(["preparing", "running"]);

  function say(text, bad = false) {
    statusLine.textContent = text || "";
    statusLine.style.color = bad ? "#b3261e" : "";
  }
  function fmtTime(seconds) {
    if (seconds == null) return "—";
    const s = Math.round(seconds), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h} ساعت و ${m} دقیقه` : m ? `${m} دقیقه و ${s % 60} ثانیه` : `${s} ثانیه`;
  }
  async function call(method, url, body) {
    const options = { method };
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    const response = await apiFetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
    return data;
  }
  function button(label, handler, primary = false) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "settings-button" + (primary ? " primary" : "");
    el.textContent = label;
    el.addEventListener("click", handler);
    return el;
  }
  async function act(id, action, payload) {
    try { await call("POST", `/api/tasks/${id}/${action}`, payload || {}); say(""); } catch (error) { say(error.message, true); }
    refresh();
  }
  async function download(id, format, partial = false) {
    const response = await apiFetch(`/api/tasks/${id}/download?format=${format}${partial ? "&partial=1" : ""}`);
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (response.status === 409 && !partial && confirm(`${data.error}\n\nنسخهٔ ناقص (با علامت‌گذاری بخش‌های جاافتاده) ساخته شود؟`)) return download(id, format, true);
      say(data.error || "ساخت فایل ممکن نشد.", true);
      return;
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = /filename\*=UTF-8''([^;]+)/.exec(disposition);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = match ? decodeURIComponent(match[1]) : `output.${format}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  }

  function render(tasks) {
    list.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("div");
      empty.className = "task-meta";
      empty.textContent = "هنوز کاری ساخته نشده است.";
      list.appendChild(empty);
      return;
    }
    for (const task of tasks) {
      const item = document.createElement("div");
      item.className = "task-item";
      const head = document.createElement("div");
      head.className = "task-head";
      const title = document.createElement("span");
      title.className = "task-title";
      title.textContent = task.title;
      title.title = task.title;
      const state = document.createElement("span");
      state.className = "task-state " + task.status;
      state.textContent = STATE_LABEL[task.status] || task.status;
      head.append(title, state);
      const bar = document.createElement("div");
      bar.className = "task-bar";
      const fill = document.createElement("span");
      fill.style.width = `${task.percent}%`;
      bar.appendChild(fill);
      const meta = document.createElement("div");
      meta.className = "task-meta";
      meta.textContent = `${task.done} از ${task.total} بخش (${task.percent}٪) · ` +
        `${task.failed ? task.failed + " ناموفق · " : ""}زمان: ${fmtTime(task.elapsed_seconds)}` +
        (task.eta_seconds != null ? ` · باقی‌مانده: ${fmtTime(task.eta_seconds)}` : "");
      item.append(head, bar, meta);
      const problem = task.error || (task.last_chunk_error && `بخش ${task.last_chunk_error.idx + 1}: ${task.last_chunk_error.error}`);
      if (problem) {
        const err = document.createElement("div");
        err.className = "task-error";
        err.textContent = problem;
        item.appendChild(err);
      }
      const actions = document.createElement("div");
      actions.className = "task-actions";
      if (ACTIVE.has(task.status)) {
        actions.append(button("توقف موقت", () => act(task.id, "pause")), button("لغو", () => act(task.id, "cancel")));
      } else {
        if (task.status !== "completed" && task.resumable) actions.append(button("ادامه", () => act(task.id, "resume"), true));
        if (task.failed) actions.append(button("تلاش دوباره برای ناموفق‌ها", () => act(task.id, "retry")));
        if (task.done > 0) for (const fmt of ["docx", "pdf", "md", "txt"]) actions.append(button(fmt.toUpperCase(), () => download(task.id, fmt)));
        actions.append(button("حذف", () => { if (confirm("این کار و نتایجش حذف شود؟")) act(task.id, "delete"); }));
      }
      item.appendChild(actions);
      list.appendChild(item);
    }
  }

  async function refresh() {
    try {
      const data = await call("GET", "/api/tasks");
      render(data.tasks || []);
      const busy = (data.tasks || []).some((t) => ACTIVE.has(t.status));
      clearTimeout(timer);
      if (!overlay.hidden) timer = setTimeout(refresh, busy ? 1500 : 6000);
    } catch (error) {
      say(error.message, true);
    }
  }

  startButton.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    const model = document.getElementById("modelSelect").value;
    if (!file) return say("ابتدا یک فایل انتخاب کن.", true);
    if (!model) return say("ابتدا یک مدل از بالای صفحه انتخاب کن.", true);
    startButton.disabled = true;
    say("در حال ارسال و تقسیم فایل…");
    const options = {
      kind: document.getElementById("taskKind").value, model, target: document.getElementById("taskTarget").value,
      style: document.getElementById("taskStyle").value.trim(),
    };
    try {
      const response = await apiFetch("/api/tasks", {
        method: "POST", body: file,
        headers: { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name),
                   "X-Task-Options": encodeURIComponent(JSON.stringify(options)) },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `خطا (${response.status})`);
      fileInput.value = "";
      say(`کار ساخته شد: ${data.task.total} بخش.`);
      refresh();
    } catch (error) {
      say(error.message, true);
    } finally {
      startButton.disabled = false;
    }
  });

  document.getElementById("tasksButton").addEventListener("click", () => { overlay.hidden = false; say(""); refresh(); });
  const close = () => { overlay.hidden = true; clearTimeout(timer); };
  document.getElementById("closeTasks").addEventListener("click", close);
  overlay.addEventListener("click", (event) => { if (event.target === overlay) close(); });
})();
