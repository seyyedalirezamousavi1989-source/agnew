/* LocalChat extras: file attachments, model manager, search settings.
   Loaded after app.js and shares its globals (showToast, loadModels, capsOf …). */

const attachButton = document.getElementById("attachButton");
const fileInput = document.getElementById("fileInput");
const attachmentTray = document.getElementById("attachmentTray");
const dropHint = document.getElementById("dropHint");

const IMAGE_MAX_SIDE = 1280;
const AUDIO_EXTENSIONS = /\.(mp3|wav|m4a|aac|ogg|oga|opus|flac|webm|weba|mp4)$/i;
const MAX_PENDING = 8;
let pendingAttachments = [];
let attachmentCounter = 0;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value >= 100 || unit === 0 ? Math.round(value) : value.toFixed(1)} ${units[unit]}`;
}

function pendingAttachmentCount() {
  return pendingAttachments.filter((item) => item.status === "ready").length;
}
function attachmentsUploading() {
  return pendingAttachments.some((item) => item.status === "uploading");
}
function getPendingAttachments() {
  return pendingAttachments.filter((item) => item.status === "ready");
}
function clearPendingAttachments() {
  pendingAttachments = [];
  renderTray();
  updateChatAccess();
}

const KIND_ICONS = { image: "🖼", text: "📄", archive: "🗜", document: "📑", audio: "🎧" };

function buildChip(item, removable) {
  const chip = document.createElement("span");
  chip.className = `attachment-chip ${item.status || ""}`.trim();
  if (item.thumb) {
    const image = document.createElement("img");
    image.src = item.thumb;
    image.alt = "";
    chip.append(image);
  } else {
    const icon = document.createElement("span");
    icon.className = "chip-icon";
    icon.textContent = KIND_ICONS[item.kind] || "📎";
    chip.append(icon);
  }
  const text = document.createElement("span");
  text.className = "chip-text";
  const name = document.createElement("span");
  name.className = "chip-name";
  name.textContent = item.name;
  name.title = item.name;
  text.append(name);
  const sub = document.createElement("span");
  sub.className = "chip-sub";
  sub.textContent = item.error || item.sub || "";
  if (sub.textContent) text.append(sub);
  chip.append(text);
  if (removable) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chip-remove";
    remove.setAttribute("aria-label", "حذف پیوست");
    remove.textContent = "×";
    remove.addEventListener("click", () => removePending(item.key));
    chip.append(remove);
  }
  return chip;
}

function renderAttachmentChips(attachments) {
  const wrap = document.createElement("div");
  wrap.className = "message-attachments";
  attachments.forEach((attachment) => wrap.append(buildChip({ ...attachment, status: "ready" }, false)));
  return wrap;
}

function renderTray() {
  attachmentTray.replaceChildren();
  attachmentTray.hidden = pendingAttachments.length === 0;
  pendingAttachments.forEach((item) => attachmentTray.append(buildChip(item, true)));
}

function removePending(key) {
  const item = pendingAttachments.find((entry) => entry.key === key);
  pendingAttachments = pendingAttachments.filter((entry) => entry.key !== key);
  if (item && item.id) {
    fetch("/api/attachments/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: item.id }),
    }).catch(() => {});
  }
  renderTray();
  updateChatAccess();
}

/* ---------- client-side conversion (no libraries needed) ---------- */

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

async function prepareImage(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, IMAGE_MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, width, height);
  context.drawImage(bitmap, 0, 0, width, height);
  const blob = await canvasToBlob(canvas, "image/jpeg", 0.88);
  const thumbScale = 64 / Math.max(width, height);
  const thumbCanvas = document.createElement("canvas");
  thumbCanvas.width = Math.max(1, Math.round(width * thumbScale));
  thumbCanvas.height = Math.max(1, Math.round(height * thumbScale));
  thumbCanvas.getContext("2d").drawImage(bitmap, 0, 0, thumbCanvas.width, thumbCanvas.height);
  if (bitmap.close) bitmap.close();
  return { blob, thumb: thumbCanvas.toDataURL("image/jpeg", 0.6) };
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => { for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i)); };
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function audioToWav(file) {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx || !window.OfflineAudioContext) throw new Error("این مرورگر از پردازش صدا پشتیبانی نمی‌کند.");
  const context = new AudioCtx();
  let decoded;
  try {
    decoded = await context.decodeAudioData(await file.arrayBuffer());
  } catch (error) {
    throw new Error("قالب این فایل صوتی خوانده نشد.");
  } finally {
    context.close();
  }
  const seconds = Math.min(decoded.duration, 600);
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(seconds * 16000)), 16000);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return { blob: encodeWav(rendered.getChannelData(0), 16000), seconds: decoded.duration };
}

/* ---------- upload ---------- */

async function uploadBlob(blob, name) {
  const response = await fetch("/api/upload", {
    method: "POST",
    headers: { "X-Filename": encodeURIComponent(name), "Content-Type": "application/octet-stream" },
    body: blob,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `آپلود ناموفق بود (${response.status}).`);
  return data.attachment;
}

function describeAttachment(info) {
  if (info.kind === "audio") return `${Math.round(info.audio_seconds)} ثانیه صدا`;
  if (info.kind === "image") return formatBytes(info.size);
  if (info.chars) return `${formatBytes(info.size)} · ${info.chars.toLocaleString("fa-IR")} نویسه`;
  return formatBytes(info.size);
}

async function addOneFile(file) {
  const item = {
    key: `a${attachmentCounter += 1}`, name: file.name || "file", kind: "text",
    status: "uploading", sub: formatBytes(file.size), thumb: "", id: "", error: "",
  };
  pendingAttachments.push(item);
  renderTray();
  updateChatAccess();
  try {
    let blob = file;
    const isImage = file.type.startsWith("image/");
    const isAudio = file.type.startsWith("audio/") || (!file.type.startsWith("video/") && AUDIO_EXTENSIONS.test(file.name) && file.type !== "");
    if (isImage) {
      item.kind = "image";
      try {
        const prepared = await prepareImage(file);
        blob = prepared.blob;
        item.thumb = prepared.thumb;
        item.name = item.name.replace(/\.[^.]+$/, "") + ".jpg";
      } catch (error) {
        if (file.size > 12 * 1024 * 1024) throw new Error("این تصویر خوانده یا کوچک نشد.");
      }
    } else if (isAudio) {
      item.kind = "audio";
      item.sub = "در حال تبدیل صدا…";
      renderTray();
      const converted = await audioToWav(file);
      blob = converted.blob;
      item.name = item.name.replace(/\.[^.]+$/, "") + ".wav";
    }
    const info = await uploadBlob(blob, item.name);
    Object.assign(item, { id: info.id, kind: info.kind, status: "ready", sub: describeAttachment(info) });
    if (info.notes && info.notes.length) item.sub += " · " + info.notes[0];
  } catch (error) {
    item.status = "error";
    item.error = error.message;
    setTimeout(() => {
      pendingAttachments = pendingAttachments.filter((entry) => entry !== item);
      renderTray();
      updateChatAccess();
    }, 6000);
  }
  renderTray();
  updateChatAccess();
}

async function addFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const room = MAX_PENDING - pendingAttachments.length;
  if (files.length > room) showToast(`حداکثر ${MAX_PENDING} فایل در هر پیام.`);
  await Promise.all(files.slice(0, Math.max(room, 0)).map(addOneFile));
}

/* Send time: pick/require the right model for the attached files. */
function installedVisionModel() {
  return Array.from(modelSelect.options).map((option) => option.value)
    .find((name) => name && capsOf(name).includes("vision"));
}
function installedAudioModel() {
  return Array.from(modelSelect.options).map((option) => option.value)
    .find((name) => name && capsOf(name).includes("audio"));
}

function prepareModelForAttachments(attachments) {
  if (attachments.some((item) => item.kind === "image") && !capsOf(modelSelect.value).includes("vision")) {
    const vision = installedVisionModel();
    if (!vision) {
      showToast("برای دیدن عکس یک مدل بینایی لازم است؛ یکی را دانلود کن.");
      openModelManager("vision");
      return false;
    }
    modelSelect.value = vision;
    try { localStorage.setItem(MODEL_KEY, vision); } catch (error) { /* ignore */ }
    updateThinkButton();
    showToast(`برای تحلیل عکس به مدل بینایی ${vision} تغییر کردم.`);
  }
  if (attachments.some((item) => item.kind === "audio") && !installedAudioModel()) {
    showToast("برای فایل صوتی یک مدل صوتی لازم است؛ gemma4 را دانلود کن.");
    openModelManager("audio");
    return false;
  }
  return true;
}

attachButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  addFiles(fileInput.files);
  fileInput.value = "";
});
messageInput.addEventListener("paste", (event) => {
  const files = Array.from(event.clipboardData?.files || []);
  if (files.length) {
    event.preventDefault();
    addFiles(files);
  }
});
let dragDepth = 0;
function hasFiles(event) {
  return Array.from(event.dataTransfer?.types || []).includes("Files");
}
document.addEventListener("dragenter", (event) => {
  if (!hasFiles(event)) return;
  dragDepth += 1;
  dropHint.hidden = false;
  document.body.classList.add("drag-over");
});
document.addEventListener("dragover", (event) => { if (hasFiles(event)) event.preventDefault(); });
document.addEventListener("dragleave", (event) => {
  if (!hasFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) { dropHint.hidden = true; document.body.classList.remove("drag-over"); }
});
document.addEventListener("drop", (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  dropHint.hidden = true;
  document.body.classList.remove("drag-over");
  if (!attachButton.disabled) addFiles(event.dataTransfer.files);
});

/* ---------- model manager ---------- */

const modelsOverlay = document.getElementById("modelsOverlay");
const modelsList = document.getElementById("modelsList");
const modelsStatus = document.getElementById("modelsStatus");
const modelsTabs = document.getElementById("modelsTabs");
const customModelInput = document.getElementById("customModel");
const pullCustomButton = document.getElementById("pullCustom");
let modelsTab = "vision";
let catalogItems = [];
let installedItems = [];
let pullController = null;
let pullingModel = "";

function setModelsStatus(text, kind = "") {
  modelsStatus.className = "settings-status" + (kind ? " " + kind : "");
  modelsStatus.textContent = text;
}

function modelCard(entry) {
  const card = document.createElement("div");
  card.className = "model-item";
  card.dataset.model = entry.id;
  const head = document.createElement("div");
  head.className = "model-head";
  const name = document.createElement("span");
  name.className = "model-name";
  name.textContent = entry.id;
  const tags = document.createElement("span");
  tags.className = "model-tags";
  const addTag = (text, cls = "") => {
    const tag = document.createElement("span");
    tag.className = `model-tag ${cls}`.trim();
    tag.textContent = text;
    tags.append(tag);
  };
  if (entry.recommended) addTag("پیشنهادی", "rec");
  if (entry.vision) addTag("بینایی 👁");
  if (entry.audio) addTag("صوتی 🎧");
  if (entry.thinking) addTag("تفکر");
  head.append(name, tags);
  card.append(head);
  if (entry.desc) {
    const desc = document.createElement("div");
    desc.className = "model-desc";
    desc.textContent = entry.desc;
    card.append(desc);
  }
  const actions = document.createElement("div");
  actions.className = "model-actions";
  const size = document.createElement("span");
  size.className = "model-size";
  size.textContent = entry.size || "";
  actions.append(size);
  if (pullingModel === entry.id) {
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "settings-button danger";
    cancel.textContent = "لغو دانلود";
    cancel.addEventListener("click", () => pullController && pullController.abort());
    actions.append(cancel);
  } else if (entry.installed) {
    const installed = document.createElement("span");
    installed.className = "model-tag";
    installed.textContent = "نصب شده ✓";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "settings-button danger";
    remove.textContent = "حذف";
    remove.addEventListener("click", () => deleteModel(entry.id));
    actions.append(installed, remove);
  } else {
    const pull = document.createElement("button");
    pull.type = "button";
    pull.className = "settings-button primary";
    pull.textContent = "دانلود";
    pull.disabled = Boolean(pullingModel);
    pull.addEventListener("click", () => pullModel(entry.id));
    actions.append(pull);
  }
  card.append(actions);
  if (pullingModel === entry.id) {
    const bar = document.createElement("div");
    bar.className = "model-progress";
    bar.append(document.createElement("div"));
    const text = document.createElement("div");
    text.className = "model-progress-text";
    card.append(bar, text);
  }
  return card;
}

function renderModelsList() {
  modelsTabs.querySelectorAll("button").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === modelsTab));
  modelsList.replaceChildren();
  let items;
  if (modelsTab === "installed") {
    items = installedItems;
  } else {
    items = catalogItems.filter((entry) => entry.group === modelsTab);
  }
  if (!items.length) {
    modelsList.textContent = modelsTab === "installed" ? "هنوز مدلی نصب نیست." : "موردی نیست.";
    return;
  }
  items.forEach((entry) => modelsList.append(modelCard(entry)));
}

async function refreshModelsData(keepStatus = false) {
  try {
    const response = await fetch("/api/models/catalog");
    const data = await response.json();
    catalogItems = data.models || [];
    if (!data.ollama_up) setModelsStatus("Ollama در دسترس نیست؛ ابتدا آن را اجرا کن.", "error");
    else if (!pullingModel && !keepStatus) setModelsStatus(data.ollama_version ? `نسخهٔ Ollama: ${data.ollama_version}` : "");
  } catch (error) {
    setModelsStatus("دریافت فهرست مدل‌ها ناموفق بود.", "error");
  }
  try {
    const response = await fetch("/api/models/capabilities");
    if (response.ok) {
      const data = await response.json();
      modelCaps = data.capabilities || modelCaps;
      installedItems = Object.keys(modelCaps).sort().map((id) => {
        const caps = modelCaps[id] || [];
        return { id, installed: true, vision: caps.includes("vision"), audio: caps.includes("audio"),
                 thinking: caps.includes("thinking"), desc: caps.includes("embedding") && !caps.includes("completion") ? "مدل embedding (برای چت نیست)" : "" };
      });
    }
  } catch (error) {
    installedItems = [];
  }
  renderModelsList();
}

function openModelManager(tab = "vision") {
  modelsTab = tab;
  modelsOverlay.hidden = false;
  renderModelsList();
  refreshModelsData();
}

function updateProgress(model, state, line) {
  const card = modelsList.querySelector(`[data-model="${CSS.escape(model)}"]`);
  const bar = card && card.querySelector(".model-progress > div");
  const text = card && card.querySelector(".model-progress-text");
  if (line.digest && line.total) {
    state.layers[line.digest] = { total: line.total, completed: line.completed || 0 };
  }
  const layers = Object.values(state.layers);
  const total = layers.reduce((sum, layer) => sum + layer.total, 0);
  const completed = layers.reduce((sum, layer) => sum + layer.completed, 0);
  const percent = total ? Math.min(100, Math.round((completed / total) * 100)) : 0;
  if (bar) bar.style.width = `${percent}%`;
  if (text) text.textContent = total ? `${percent}% — ${formatBytes(completed)} / ${formatBytes(total)}` : (line.status || "");
}

async function pullModel(model) {
  model = (model || "").trim();
  if (!model || pullingModel) return;
  pullingModel = model;
  pullController = new AbortController();
  setModelsStatus(`در حال دانلود ${model} … (می‌توانی پنجره را ببندی؛ دانلود ادامه دارد)`);
  renderModelsList();
  const state = { layers: {} };
  let ok = false;
  try {
    const response = await fetch("/api/models/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
      signal: pullController.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `دانلود ناموفق بود (${response.status}).`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = done ? "" : lines.pop();
      for (const raw of lines) {
        if (!raw.trim()) continue;
        const line = JSON.parse(raw);
        if (line.error) throw new Error(line.error);
        updateProgress(model, state, line);
        if (line.status === "success") ok = true;
      }
      if (done) break;
    }
    if (!ok) throw new Error("دانلود کامل نشد.");
    setModelsStatus(`${model} نصب شد.`);
    showToast(`${model} نصب شد.`);
  } catch (error) {
    if (error.name === "AbortError") setModelsStatus("دانلود لغو شد؛ دفعهٔ بعد از همان‌جا ادامه پیدا می‌کند.");
    else setModelsStatus(error.message, "error");
  } finally {
    pullingModel = "";
    pullController = null;
    await refreshModelsData(true);
    if (ok) await loadModels();
  }
}

async function deleteModel(model) {
  if (!window.confirm(`مدل ${model} حذف شود؟`)) return;
  try {
    const response = await fetch("/api/models/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "حذف ناموفق بود.");
    showToast(`${model} حذف شد.`);
    await refreshModelsData();
    await loadModels();
  } catch (error) {
    setModelsStatus(error.message, "error");
  }
}

document.getElementById("modelsButton").addEventListener("click", () => openModelManager(modelsTab));
document.getElementById("closeModels").addEventListener("click", () => { modelsOverlay.hidden = true; });
modelsOverlay.addEventListener("click", (event) => { if (event.target === modelsOverlay) modelsOverlay.hidden = true; });
modelsTabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-tab]");
  if (!button) return;
  modelsTab = button.dataset.tab;
  renderModelsList();
});
pullCustomButton.addEventListener("click", () => {
  const name = customModelInput.value.trim();
  if (!name) return;
  customModelInput.value = "";
  pullModel(name);
});
customModelInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); pullCustomButton.click(); }
});

/* ---------- search settings (inside the settings dialog) ---------- */

const searchEngineSelect = document.getElementById("searchEngine");
const serperKeyInput = document.getElementById("serperKey");
const braveKeyInput = document.getElementById("braveKey");
const googleKeyInput = document.getElementById("googleKey");
const googleCxInput = document.getElementById("googleCx");
const searxngUrlInput = document.getElementById("searxngUrl");
const searchStatus = document.getElementById("searchStatus");

function showSearchSettings(search) {
  if (!search) return;
  searchEngineSelect.value = search.engine;
  serperKeyInput.value = "";
  braveKeyInput.value = "";
  googleKeyInput.value = "";
  serperKeyInput.placeholder = search.has_serper_key ? `Serper ذخیره شده (${search.serper_hint}) — برای تغییر، کلید تازه بنویس` : "Serper API key";
  braveKeyInput.placeholder = search.has_brave_key ? `Brave ذخیره شده (${search.brave_hint}) — برای تغییر، کلید تازه بنویس` : "Brave Search API key";
  googleKeyInput.placeholder = search.google_api_hint ? `Google key ذخیره شده (${search.google_api_hint})` : "Google API key (اختیاری)";
  googleCxInput.value = search.google_cx || "";
  searxngUrlInput.value = search.searxng_url || "";
  setSearchEngineLabel(search.engine);
  const parts = [];
  if (search.engine === "auto") parts.push("حالت خودکار: ترکیب موتورهای در دسترس");
  if (search.has_serper_key) parts.push("Serper فعال");
  if (search.has_brave_key) parts.push("Brave فعال");
  if (search.has_searxng) parts.push("SearXNG تنظیم شده");
  if (!search.has_serper_key && !search.has_brave_key && !search.has_google_cse) {
    parts.push("بدون کلید: جست‌وجوی رایگان با پوشش محدودتر");
  }
  searchStatus.className = "settings-status";
  searchStatus.textContent = parts.join(" · ");
}

async function loadSearchSettings() {
  try {
    const response = await fetch("/api/settings");
    const data = await response.json();
    showSearchSettings(data.search);
  } catch (error) {
    searchStatus.className = "settings-status error";
    searchStatus.textContent = "دریافت تنظیمات جست‌وجو ناموفق بود.";
  }
}

document.getElementById("saveSearch").addEventListener("click", async () => {
  const body = { engine: searchEngineSelect.value, google_cx: googleCxInput.value.trim(), searxng_url: searxngUrlInput.value.trim() };
  if (serperKeyInput.value.trim()) body.serper_api_key = serperKeyInput.value.trim();
  if (braveKeyInput.value.trim()) body.brave_api_key = braveKeyInput.value.trim();
  if (googleKeyInput.value.trim()) body.google_api_key = googleKeyInput.value.trim();
  try {
    const response = await fetch("/api/settings/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "ذخیره ناموفق بود.");
    showSearchSettings(data.search);
    showToast("تنظیمات جست‌وجو ذخیره شد.");
  } catch (error) {
    searchStatus.className = "settings-status error";
    searchStatus.textContent = error.message;
  }
});
document.getElementById("clearSearchKeys").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/settings/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ serper_api_key: "", brave_api_key: "", google_api_key: "", google_cx: "" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "پاک کردن ناموفق بود.");
    showSearchSettings(data.search);
    showToast("کلیدها پاک شدند.");
  } catch (error) {
    searchStatus.className = "settings-status error";
    searchStatus.textContent = error.message;
  }
});
document.getElementById("settingsButton").addEventListener("click", loadSearchSettings);
fetch("/api/settings").then((r) => r.json()).then((d) => setSearchEngineLabel(d.search && d.search.engine)).catch(() => {});

updateChatAccess();

/* ---------------- About / version panel + voice diagnostics (2.0.5) ----------------
   UI_VERSION is the version of THESE static files. The server reports its own version at /api/status.
   When the two differ you are looking at new files served by an old, still-running backend (or the reverse):
   the badge turns red and the panel says so. test_version.py keeps UI_VERSION equal to version.py. */
const UI_VERSION = "2.0.5";
const aboutOverlay = document.getElementById("aboutOverlay");
const versionBadge = document.getElementById("versionBadge");
let lastAboutText = "";

function formatUptime(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} ساعت و ${m} دقیقه` : `${m} دقیقه و ${s % 60} ثانیه`;
}

async function loadAbout() {
  const title = document.getElementById("aboutVersion");
  const warn = document.getElementById("aboutWarn");
  const grid = document.getElementById("aboutGrid");
  const changes = document.getElementById("aboutChanges");
  let info = null;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    info = await response.json();
  } catch (_) { /* handled below */ }
  const dot = document.getElementById("versionDot");
  if (!info || !info.version) {
    versionBadge.className = "version-badge bad";
    document.getElementById("versionBadgeText").textContent = "سرور در دسترس نیست";
    title.textContent = "—";
    warn.hidden = false; warn.textContent = "به سرور محلی وصل نشدم؛ برنامه را دوباره اجرا کن.";
    return;
  }
  const mismatch = info.version !== UI_VERSION;
  versionBadge.className = "version-badge " + (mismatch ? "bad" : "ok");
  document.getElementById("versionBadgeText").textContent = `نسخه ${info.version}` + (mismatch ? " (ناهماهنگ)" : "");
  title.textContent = `LocalChat ${info.version}`;
  warn.hidden = !mismatch;
  if (mismatch) {
    warn.textContent = `رابط کاربری نسخهٔ ${UI_VERSION} است ولی سرور در حال اجرا نسخهٔ ${info.version} است. ` +
      "یعنی هنوز یک پردازش قدیمی LocalChat/Python باز است. همهٔ پنجره‌های برنامه و پردازش‌های python/LocalChat را ببند " +
      "(Task Manager) و دوباره اجرا کن.";
  }
  const started = info.started_at ? new Date(info.started_at * 1000).toLocaleString("fa-IR") : "—";
  const rows = [
    ["نسخهٔ سرور", info.version], ["نسخهٔ رابط کاربری", UI_VERSION],
    ["نام ساخت", info.build || "—"], ["تاریخ ساخت", info.build_date || "—"],
    ["نوع اجرا", info.frozen ? "نسخهٔ نصبی/EXE" : "اجرای مستقیم با Python"],
    ["شناسهٔ پردازش (PID)", info.pid], ["شروع سرور", started], ["مدت اجرا", formatUptime(info.uptime_seconds)],
    ["مسیر کد", info.code_dir || "—"], ["مفسر/برنامه", info.executable || "—"], ["Python", info.python || "—"],
  ];
  grid.replaceChildren();
  rows.forEach(([name, value]) => {
    const dt = document.createElement("dt"); dt.textContent = name;
    const dd = document.createElement("dd"); dd.textContent = String(value);
    grid.append(dt, dd);
  });
  changes.replaceChildren();
  (info.changelog || []).forEach((line) => { const li = document.createElement("li"); li.textContent = line; changes.append(li); });
  lastAboutText = rows.map(([name, value]) => `${name}: ${value}`).join("\n");
}

versionBadge?.addEventListener("click", () => { aboutOverlay.hidden = false; loadAbout(); });
document.getElementById("closeAbout")?.addEventListener("click", () => { aboutOverlay.hidden = true; });
aboutOverlay?.addEventListener("click", (event) => { if (event.target === aboutOverlay) aboutOverlay.hidden = true; });
document.getElementById("refreshAbout")?.addEventListener("click", loadAbout);
document.getElementById("copyAbout")?.addEventListener("click", async () => {
  const status = document.getElementById("aboutStatus");
  try { await navigator.clipboard.writeText(lastAboutText || "LocalChat " + UI_VERSION); status.textContent = "اطلاعات نسخه کپی شد."; }
  catch (_) { status.textContent = "دسترسی کپی در مرورگر فراهم نیست."; }
});
loadAbout();   // keeps the sidebar badge truthful from the first second

document.getElementById("diagnoseTts")?.addEventListener("click", async () => {
  const status = document.getElementById("ttsStatus");
  status.textContent = "در حال بررسی…";
  try {
    const response = await fetch("/api/voice/diagnose", { cache: "no-store" });
    const data = await response.json();
    const list = document.createElement("ul");
    list.className = "diag-list";
    (data.checks || []).forEach((check) => {
      const li = document.createElement("li");
      li.className = check.ok ? "ok" : "fail";
      li.textContent = check.name;
      if (check.detail) { const small = document.createElement("small"); small.textContent = check.detail; li.append(small); }
      list.append(li);
    });
    status.replaceChildren(document.createTextNode(`موتور فعلی: ${data.provider} · صدا: ${data.voice}`), list);
  } catch (error) {
    status.textContent = "عیب‌یابی انجام نشد: " + error.message;
  }
});
