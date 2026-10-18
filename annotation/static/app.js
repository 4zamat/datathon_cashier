"use strict";

const $ = id => document.getElementById(id);
const player = $("player");
const overlay = $("box-overlay");
const labels = {person: "Человек", cash: "Деньги", hand: "Кисть руки", phone: "Телефон", tray: "Лоток", storage: "Сейф / хранение", laptop: "Ноутбук", cash_counter: "Счётчик банкнот", banknote_detector: "Детектор банкнот", bag: "Сумка / пакет", other: "Другое"};
const colors = {person: "#fb6979", cash: "#e6c15a", hand: "#eea269", phone: "#74bedf", tray: "#bcc5ed", storage: "#69d990", laptop: "#f3d351", cash_counter: "#6b9dff", banknote_detector: "#c48aff", bag: "#e29ab8", other: "#dbdbe3"};
const assessments = {uncertain: "Неясно", possible_violation: "Есть признаки", no_violation: "Нарушений не видно"};
const priority = new Set([21, 23, 30, 38, 48]);
let config, videos = [], current, doc, frameId = null, boxId = null, editId = "", mode = "draw";
let dirty = false, serial = 0, saveTimer, saveChain = Promise.resolve(), switching = false, drag = null;

const escapeHTML = text => String(text).replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
const clip = (n, min, max) => Math.max(min, Math.min(max, n));
const timestamp = seconds => {
  const ms = Math.round(Math.max(0, seconds || 0) * 1000);
  return `${String(Math.floor(ms / 60000)).padStart(2, "0")}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
};
const currentFrame = () => doc?.frames.find(frame => frame.id === frameId);
const currentBox = () => currentFrame()?.boxes.find(box => box.id === boxId);

function message(text = "") {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {
    "Content-Type": "application/json", "X-Annotation-Token": config?.token || "", ...options.headers,
  }});
  if (!response.ok) {
    let text = `Ошибка ${response.status}`;
    try {
      const error = await response.json();
      text = Array.isArray(error.detail) ? error.detail.map(item => item.msg).join("; ") : error.detail || text;
    } catch (_) { /* Keep HTTP status when the server did not return JSON. */ }
    throw new Error(text);
  }
  return response;
}

function savedStatus(text, className = "") {
  $("save-status").textContent = text;
  $("save-status").className = className;
}

function changed() {
  if (!doc) return;
  dirty = true;
  serial++;
  savedStatus("Есть изменения…", "pending");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save().catch(error => message(error.message)), 900);
}

function save() {
  clearTimeout(saveTimer);
  const run = async () => {
    if (!doc || !dirty) return;
    const version = serial;
    const snapshot = structuredClone(doc);
    savedStatus("Сохраняем…", "pending");
    try {
      const result = await (await api(`/api/videos/${snapshot.video_id}/annotation`, {
        method: "PUT", body: JSON.stringify(snapshot),
      })).json();
      doc.revision = result.revision;
      doc.updated_at = result.updated_at;
      dirty = version !== serial;
      savedStatus(dirty ? "Есть изменения…" : "Сохранено на компьютере", dirty ? "pending" : "");
      current.events = doc.events.length;
      current.frames = doc.frames.length;
      current.status = doc.watched_full ? "reviewed" : "draft";
      renderVideos();
    } catch (error) {
      savedStatus("Не сохранено", "error");
      throw error;
    }
  };
  saveChain = saveChain.then(run, run);
  return saveChain;
}

function needReviewer() {
  if (doc?.reviewer.trim()) return true;
  message("Сначала укажите имя в поле «Кто размечает» ниже.");
  $("reviewer").focus();
  return false;
}

function renderVideos() {
  const search = $("search").value.toLocaleLowerCase().replace(/х/g, "x");
  const filter = $("video-filter").value;
  const visible = videos.filter(item => item.name.toLocaleLowerCase().replace(/х/g, "x").includes(search))
    .filter(item => filter === "all" || (filter === "priority" ? priority.has(Number(item.name.match(/\d+/)?.[0])) : item.status === filter));
  $("video-list").innerHTML = visible.map(item => `<button class="video-item ${current?.id === item.id ? "active" : ""}" data-video="${item.id}" ${current?.id === item.id ? 'aria-current="true"' : ""}><span class="video-symbol">▶</span><span><span class="item-name">${escapeHTML(item.name)}</span><small>${Math.round(item.duration_sec)} с · ${item.events} соб. · ${item.frames} кадр.</small></span><span class="status-dot ${item.status}" title="${{new:"Не начато",draft:"В работе",reviewed:"Просмотрено целиком",error:"Ошибка сохранённых данных"}[item.status]}"></span></button>`).join("") || '<p class="empty">Нет записей</p>';
  const watched = videos.filter(item => item.status === "reviewed").length;
  $("video-count").textContent = videos.length;
  $("progress-text").textContent = `${watched} из ${videos.length} просмотрено целиком`;
  $("progress").max = Math.max(videos.length, 1);
  $("progress").value = watched;
}

async function selectVideo(id) {
  if (switching || current?.id === id) return;
  switching = true;
  document.querySelector(".main-area").inert = true;
  try {
    await save();
    const next = videos.find(item => item.id === id);
    const loaded = await (await api(`/api/videos/${id}/annotation`)).json();
    player.pause();
    current = next;
    doc = loaded;
    dirty = false;
    frameId = doc.frames.at(-1)?.id || null;
    boxId = null;
    if (!doc.reviewer) doc.reviewer = localStorage.getItem("cashier.reviewer") || "";
    $("reviewer").value = doc.reviewer;
    $("scene-group").value = doc.scene_group_id;
    $("notes").value = doc.notes;
    $("watched").checked = doc.watched_full;
    $("layout-info").textContent = doc.layout_group_id ? `${doc.layout_group_id}: ${doc.layout_notes} Это расстановка оборудования, а не группа эпизодов для обучения.` : "";
    $("video-title").textContent = current.name;
    $("video-meta").textContent = `${current.width} × ${current.height} · ${current.duration_sec.toFixed(2)} с · Исходная запись`;
    $("event-start").max = current.duration_sec;
    $("event-end").max = current.duration_sec;
    player.src = `/media/${id}`;
    player.load();
    restoreDraft();
    renderEvents();
    renderFrames();
    setTab("video");
    renderVideos();
    savedStatus(doc.revision ? "Сохранено на компьютере" : "Разметка ещё не начата");
    localStorage.setItem("cashier.last-video", id);
    message();
  } catch (error) {
    message(error.message);
  } finally {
    switching = false;
    document.querySelector(".main-area").inert = false;
  }
}

function persistDraft() {
  if (!doc) return;
  doc.event_draft = {
    id: editId, start: $("event-start").value, end: $("event-end").value,
    event_type: $("event-type").value, assessment: $("assessment").value,
    observation: $("observation").value,
  };
  $("draft-status").hidden = false;
  changed();
}

function restoreDraft() {
  const draft = doc.event_draft;
  editId = draft?.id || "";
  $("event-start").value = draft?.start || "";
  $("event-end").value = draft?.end || "";
  $("observation").value = draft?.observation || "";
  $("event-type").value = draft?.event_type || "unknown";
  $("assessment").value = draft?.assessment || "uncertain";
  $("event-form-title").textContent = editId ? "Изменить событие" : "Добавить событие";
  $("add-event").textContent = editId ? "Сохранить событие" : "Добавить событие";
  $("draft-status").hidden = !draft;
  showRule();
}

function resetDraft() {
  doc.event_draft = null;
  restoreDraft();
  changed();
}

function showRule() {
  const rule = config?.rules[$("event-type").value];
  $("rule-text").textContent = rule ? `${rule.document}, ${rule.clause}. ${rule.summary}` : "Запишите наблюдаемые действия. Если данных недостаточно, оставьте оценку «Неясно».";
}

function renderEvents() {
  $("event-count").textContent = doc.events.length;
  $("events-list").innerHTML = [...doc.events].sort((a, b) => a.start_sec - b.start_sec).map(event => `<article class="event-card" data-event="${event.id}"><div class="event-top"><button class="time-link" data-action="seek">${timestamp(event.start_sec)} — ${timestamp(event.end_sec)}</button><span class="event-tag">${assessments[event.assessment]}</span></div><p>${escapeHTML(event.observation)}</p><small>${escapeHTML(config.event_names[event.event_type] || event.event_type)}</small><div class="event-buttons"><button class="text-button" data-action="edit">Изменить</button><button class="text-button" data-action="delete">Удалить</button></div></article>`).join("") || '<p class="empty">Отметьте время и добавьте первое событие</p>';
}

function setTab(tab) {
  $("video-panel").hidden = tab !== "video";
  $("frame-panel").hidden = tab !== "frame";
  $("video-tab").setAttribute("aria-selected", String(tab === "video"));
  $("frame-tab").setAttribute("aria-selected", String(tab === "frame"));
  if (tab === "frame") player.pause();
}

function setMode(value) {
  mode = value;
  $("draw-mode").className = value === "draw" ? "primary" : "secondary";
  $("select-mode").className = value === "select" ? "primary" : "secondary";
  overlay.classList.toggle("selecting", value === "select");
  $("draw-mode").setAttribute("aria-pressed", String(value === "draw"));
  $("select-mode").setAttribute("aria-pressed", String(value === "select"));
  $("drawing-help").textContent = value === "draw" ? "Добавление рамок: протяните мышью новый прямоугольник. Можно рисовать несколько подряд и поверх существующих." : "Выбор рамки: нажмите на объект или его название ниже. Для следующего объекта нажмите «+ Новая рамка».";
}

function newBox() {
  drag = null;
  boxId = null;
  setMode("draw");
  renderBoxes();
}

function renderFrames() {
  $("frame-count").textContent = doc.frames.length;
  $("frame-strip").innerHTML = doc.frames.map(frame => `<button class="frame-chip ${frame.id === frameId ? "active" : ""}" data-frame="${frame.id}">${timestamp(frame.timestamp_sec)} · ${frame.boxes.length} ${frame.reviewed ? "✓" : ""}</button>`).join("");
  const frame = currentFrame();
  $("frame-canvas").hidden = !frame;
  $("frame-empty").hidden = !!frame;
  $("frame-reviewed").disabled = !frame;
  $("delete-frame").disabled = !frame;
  $("review-profile").disabled = !frame;
  $("boxes-count").textContent = frame ? `${frame.boxes.length} рамок` : "";
  if (frame) {
    $("frame-image").src = `/frames/${doc.video_id}/${frame.id}.jpg`;
    overlay.setAttribute("viewBox", `0 0 ${frame.width} ${frame.height}`);
    $("frame-time").textContent = `Кадр ${frame.frame_index} · ${timestamp(frame.timestamp_sec)} · ${frame.width} × ${frame.height}`;
    if (frame.reviewed) {
      const scope = frame.reviewed_classes || [];
      const equipment = config.review_profiles.equipment;
      $("review-profile").value = scope.length === equipment.length && equipment.every(label => scope.includes(label)) ? "equipment" : "all";
    }
  }
  reviewStatus();
  renderBoxes();
}

function reviewStatus() {
  const frame = currentFrame();
  const scope = config.review_profiles[$("review-profile").value];
  $("frame-reviewed").checked = !!frame?.reviewed && scope.every(label => (frame.reviewed_classes || []).includes(label));
  $("frame-provenance").textContent = frame ? [frame.reviewed_by ? `Проверил: ${frame.reviewed_by}.` : "", frame.annotation_notes || ""].filter(Boolean).join(" ") : "";
}

function svg(tag, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}

function renderBoxes() {
  overlay.replaceChildren();
  const frame = currentFrame();
  if (!frame) { $("box-editor").hidden = true; $("box-list").replaceChildren(); return; }
  const display = [...frame.boxes];
  if (drag?.kind === "draw") display.push({...drag.box, id: "drawing", label: $("box-label").value});
  for (const box of display) {
    overlay.append(svg("rect", {x: box.x * frame.width, y: box.y * frame.height, width: box.width * frame.width,
      height: box.height * frame.height, class: `object-box ${box.id === boxId ? "selected" : ""} ${box.id === "drawing" ? "drawing" : ""}`, stroke: colors[box.label], "data-box": box.id}));
    const text = svg("text", {x: box.x * frame.width + 4, y: Math.max(16, box.y * frame.height - 5), class: "box-caption", "font-size": Math.max(12, frame.width / 32)});
    text.textContent = labels[box.label];
    overlay.append(text);
    if (box.id === boxId && mode === "select") {
      const size = frame.width / 45;
      overlay.append(svg("rect", {x: (box.x + box.width) * frame.width - size / 2, y: (box.y + box.height) * frame.height - size / 2,
        width: size, height: size, class: "resize-handle", "data-box": box.id, "data-handle": "resize"}));
    }
  }
  const selected = currentBox();
  if (!drag) $("box-list").innerHTML = frame.boxes.map((box, index) => `<button class="object-chip ${box.id === boxId ? "active" : ""}" data-select-box="${box.id}">${index + 1}. ${escapeHTML(labels[box.label])}${box.source === "assistant" ? " · ИИ" : ""}</button>`).join("");
  $("box-editor").hidden = !selected;
  if (selected) {
    $("selected-label").value = selected.label;
    $("occluded").checked = selected.occluded;
    if (document.activeElement !== $("box-comment")) $("box-comment").value = selected.comment;
  }
}

function invalidateFrame() {
  currentFrame().reviewed = false;
  currentFrame().reviewed_classes = [];
  currentFrame().reviewed_by = "";
  $("frame-reviewed").checked = false;
  changed();
}

function point(event) {
  const bounds = overlay.getBoundingClientRect();
  return {x: clip((event.clientX - bounds.left) / bounds.width, 0, 1), y: clip((event.clientY - bounds.top) / bounds.height, 0, 1)};
}

overlay.addEventListener("pointerdown", event => {
  if (event.button !== 0 || !currentFrame()) return;
  event.preventDefault();
  const at = point(event);
  if (mode === "select") {
    boxId = event.target.getAttribute("data-box");
    const box = currentBox();
    drag = box ? {kind: event.target.getAttribute("data-handle") || "move", at, original: {...box}, moved: false} : null;
  } else {
    drag = {kind: "draw", at, box: {x: at.x, y: at.y, width: 0, height: 0}};
  }
  overlay.setPointerCapture(event.pointerId);
  renderBoxes();
});

overlay.addEventListener("pointermove", event => {
  if (!drag) return;
  const at = point(event);
  if (drag.kind === "draw") {
    drag.box = {x: Math.min(at.x, drag.at.x), y: Math.min(at.y, drag.at.y), width: Math.abs(at.x - drag.at.x), height: Math.abs(at.y - drag.at.y)};
  } else {
    const box = currentBox(), original = drag.original;
    drag.moved = true;
    if (drag.kind === "resize") {
      box.width = clip(at.x - box.x, 1 / currentFrame().width, 1 - box.x);
      box.height = clip(at.y - box.y, 1 / currentFrame().height, 1 - box.y);
    } else {
      box.x = clip(original.x + at.x - drag.at.x, 0, 1 - box.width);
      box.y = clip(original.y + at.y - drag.at.y, 0, 1 - box.height);
    }
  }
  renderBoxes();
});

function finishPointer(event) {
  if (!drag) return;
  if (drag.kind === "draw") {
    const bounds = overlay.getBoundingClientRect();
    if (drag.box.width * bounds.width >= 4 && drag.box.height * bounds.height >= 4) {
      const box = {id: crypto.randomUUID(), label: $("box-label").value, ...drag.box, occluded: false, comment: "", annotator: doc.reviewer, source: "human"};
      currentFrame().boxes.push(box);
      boxId = box.id;
      invalidateFrame();
    }
  } else if (drag.moved) {
    currentBox().annotator = doc.reviewer;
    currentBox().source = "human";
    invalidateFrame();
  }
  drag = null;
  if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
  renderFrames();
}
overlay.addEventListener("pointerup", finishPointer);
overlay.addEventListener("pointercancel", event => {
  if (drag?.original && currentBox()) Object.assign(currentBox(), drag.original);
  drag = null;
  if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
  renderBoxes();
});

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

$("search").addEventListener("input", renderVideos);
$("video-filter").addEventListener("change", renderVideos);
$("video-list").addEventListener("click", event => {
  const button = event.target.closest("[data-video]");
  if (button) selectVideo(button.dataset.video);
});
for (const [id, direction] of [["previous", -1], ["next", 1]]) $(id).onclick = () => {
  const index = videos.findIndex(item => item.id === current?.id);
  if (videos[index + direction]) selectVideo(videos[index + direction].id);
};
$("save").onclick = () => save().then(() => message()).catch(error => message(error.message));
$("export").onclick = async () => {
  $("export").disabled = true;
  try {
    await save();
    const response = await api("/api/export", {method: "POST"});
    const name = response.headers.get("content-disposition")?.match(/filename="?([^";]+)"?/)?.[1] || "annotations.zip";
    download(await response.blob(), name);
    message(doc.event_draft ? "Архив скачан. Незавершённая форма сохранена как черновик JSON; добавьте событие, чтобы оно вошло в CSV." : "Архив разметки скачан. В events.csv входят события с указанной группой сцены; остальные сохранены в events_draft.csv.");
  } catch (error) { message(error.message); }
  finally { $("export").disabled = false; }
};
$("download-draft").onclick = () => { if (doc) download(new Blob([JSON.stringify(doc, null, 2)], {type: "application/json"}), `draft-${current.name}.json`); };
for (const [id, property] of [["reviewer", "reviewer"], ["scene-group", "scene_group_id"], ["notes", "notes"]]) {
  $(id).oninput = () => {
    if (!doc) return;
    doc[property] = $(id).value;
    if (id === "reviewer") localStorage.setItem("cashier.reviewer", $(id).value);
    changed();
  };
}
$("watched").onchange = () => {
  if (!doc) return;
  if (!needReviewer()) { $("watched").checked = false; return; }
  doc.watched_full = $("watched").checked;
  changed();
};

for (const id of ["event-start", "event-end", "observation", "event-type", "assessment"]) $(id).addEventListener("input", () => {
  if (id === "event-type") {
    if ($("event-type").value === "normal") $("assessment").value = "no_violation";
    if ($("event-type").value === "unknown") $("assessment").value = "uncertain";
    showRule();
  }
  persistDraft();
});
$("event-form").onsubmit = event => {
  event.preventDefault();
  if (!doc || !needReviewer()) return;
  const start = Number($("event-start").value), end = Number($("event-end").value);
  const observation = $("observation").value.trim();
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || end > current.duration_sec || !observation) {
    message("Проверьте интервал: начало ≤ конец ≤ длительность видео. Добавьте описание действия."); return;
  }
  const type = $("event-type").value, assessment = $("assessment").value;
  if (assessment === "possible_violation" && ["normal", "unknown"].includes(type)) {
    message("Для признаков нарушения выберите конкретный тип события или оставьте оценку «Неясно»."); return;
  }
  if (type === "normal" && assessment !== "no_violation") {
    message("Для нормального фрагмента выберите оценку «Нарушений не видно»."); return;
  }
  const item = {id: editId || crypto.randomUUID(), start_sec: start, end_sec: end, event_type: type, assessment, observation};
  const index = doc.events.findIndex(entry => entry.id === item.id);
  if (index < 0) doc.events.push(item); else doc.events[index] = item;
  resetDraft();
  renderEvents();
  message();
};
$("cancel-event").onclick = () => { if (doc) resetDraft(); };
$("events-list").onclick = event => {
  const button = event.target.closest("[data-action]");
  if (!button) return;
  const item = doc.events.find(entry => entry.id === button.closest("[data-event]").dataset.event);
  if (button.dataset.action === "seek") { setTab("video"); player.currentTime = item.start_sec; player.pause(); }
  if (button.dataset.action === "edit") {
    if (doc.event_draft && !confirm("Заменить текущий черновик формы выбранным событием?")) return;
    doc.event_draft = {id: item.id, start: String(item.start_sec), end: String(item.end_sec), observation: item.observation, event_type: item.event_type, assessment: item.assessment};
    restoreDraft(); changed();
  }
  if (button.dataset.action === "delete" && confirm("Удалить это событие? Предыдущая версия останется в резервной копии.")) {
    doc.events = doc.events.filter(entry => entry.id !== item.id);
    if (editId === item.id) resetDraft();
    renderEvents(); changed();
  }
};

function mark(which) {
  if (!current || !Number.isFinite(player.duration)) return;
  $(which).value = Math.min(Math.round(player.currentTime * 1000) / 1000, current.duration_sec);
  persistDraft();
}
$("mark-start").onclick = () => mark("event-start");
$("mark-end").onclick = () => mark("event-end");
const seek = delta => { if (current && Number.isFinite(player.duration)) player.currentTime = clip(player.currentTime + delta, 0, Math.min(player.duration, current.duration_sec)); };
$("back").onclick = () => seek(-0.5);
$("forward").onclick = () => seek(0.5);
$("speed").onchange = () => { player.playbackRate = Number($("speed").value); };
const updateClock = () => { $("clock").textContent = `${timestamp(player.currentTime)} / ${timestamp(current?.duration_sec)}`; };
player.addEventListener("timeupdate", updateClock);
player.addEventListener("loadedmetadata", () => { player.playbackRate = Number($("speed").value); updateClock(); });
player.addEventListener("error", () => message("Браузер не открыл видео. Проверьте наличие FFmpeg и перезапустите страницу. Исходный файл не меняется."));
$("video-tab").onclick = () => setTab("video");
$("frame-tab").onclick = () => setTab("frame");
$("capture").onclick = async () => {
  if (!doc || !needReviewer() || !Number.isFinite(player.duration)) return;
  player.pause();
  $("capture").disabled = true;
  switching = true;
  try {
    const frame = await (await api(`/api/videos/${doc.video_id}/capture`, {method: "POST", body: JSON.stringify({timestamp_sec: Math.min(player.currentTime, current.duration_sec)})})).json();
    if (!doc.frames.some(item => item.id === frame.id)) doc.frames.push(frame);
    frameId = frame.id;
    boxId = null;
    setTab("frame"); setMode("draw"); renderFrames(); changed(); message();
  } catch (error) { message(error.message); }
  finally { $("capture").disabled = false; switching = false; }
};
$("frame-strip").onclick = event => {
  const button = event.target.closest("[data-frame]");
  if (button) { frameId = button.dataset.frame; boxId = null; renderFrames(); }
};
$("draw-mode").onclick = newBox;
$("box-label").onchange = newBox;
$("select-mode").onclick = () => { setMode("select"); renderBoxes(); };
$("box-list").onclick = event => {
  const button = event.target.closest("[data-select-box]");
  if (button) { boxId = button.dataset.selectBox; setMode("select"); renderBoxes(); }
};
$("review-profile").onchange = reviewStatus;
$("frame-reviewed").onchange = () => {
  if (!currentFrame()) return;
  if ($("frame-reviewed").checked && $("review-profile").value === "all" && currentFrame().boxes.some(box => box.label === "other")) {
    $("frame-reviewed").checked = false;
    message("Перед подтверждением кадра уточните класс рамок «Другое». Кадр и рамки уже сохраняются как черновик.");
    return;
  }
  currentFrame().reviewed = $("frame-reviewed").checked;
  currentFrame().reviewed_classes = $("frame-reviewed").checked ? [...config.review_profiles[$("review-profile").value]] : [];
  currentFrame().reviewed_by = $("frame-reviewed").checked ? doc.reviewer : "";
  changed(); renderFrames();
};
for (const [id, property] of [["selected-label", "label"], ["occluded", "occluded"], ["box-comment", "comment"]]) {
  $(id).addEventListener("input", () => {
    const box = currentBox();
    if (!box) return;
    box[property] = id === "occluded" ? $(id).checked : $(id).value;
    box.annotator = doc.reviewer;
    box.source = "human";
    invalidateFrame(); renderBoxes();
  });
}
$("delete-box").onclick = () => {
  if (!currentBox()) return;
  currentFrame().boxes = currentFrame().boxes.filter(box => box.id !== boxId);
  boxId = null; invalidateFrame(); renderFrames();
};
$("delete-frame").onclick = () => {
  if (!currentFrame() || !confirm("Убрать этот кадр вместе с рамками из разметки?")) return;
  doc.frames = doc.frames.filter(frame => frame.id !== frameId);
  frameId = doc.frames[0]?.id || null; boxId = null;
  changed(); renderFrames();
};

document.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.code === "KeyS") {
    event.preventDefault(); save().catch(error => message(error.message)); return;
  }
  if (event.target.closest("input,textarea,select,button") || event.ctrlKey || event.metaKey || event.altKey || !current || switching) return;
  if (event.code === "Space" && !$("video-panel").hidden) {
    event.preventDefault(); if (player.paused) player.play().catch(error => message(error.message)); else player.pause();
  }
  if (event.code === "BracketLeft") { event.preventDefault(); mark("event-start"); }
  if (event.code === "BracketRight") { event.preventDefault(); mark("event-end"); }
  if (event.code === "ArrowLeft") { event.preventDefault(); seek(-0.5); }
  if (event.code === "ArrowRight") { event.preventDefault(); seek(0.5); }
});
window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });

async function initialize() {
  config = await (await api("/api/config")).json();
  videos = await (await api("/api/videos")).json();
  $("event-type").innerHTML = Object.entries(config.event_names).map(([key, value]) => `<option value="${key}">${escapeHTML(value)}</option>`).join("");
  $("selected-label").innerHTML = $("box-label").innerHTML;
  $("save-path").textContent = config.save_directory;
  renderVideos();
  const previous = localStorage.getItem("cashier.last-video");
  const first = videos.find(video => video.id === previous) || videos.find(video => /[XХ]-30\./i.test(video.name)) || videos[0];
  if (first) await selectVideo(first.id);
  else { savedStatus("Нет видео"); message("Положите MOV или MP4 в data/videos и перезапустите локальный сервер."); }
  if (config.scan_errors.length) message(config.scan_errors.join("; "));
}
initialize().catch(error => { savedStatus("Не удалось загрузить", "error"); message(error.message); });
