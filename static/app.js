"use strict";

const form = document.querySelector("#upload-form");
const statusNode = document.querySelector("#status");
const submitButton = document.querySelector("#submit");
const retryButton = document.querySelector("#retry");
const resultNode = document.querySelector("#result");
let activeJob = null;
let activeKey = "";
let videoUrl = null;
const player = document.querySelector("#player");
const videoInput = document.querySelector("#video");
const titles = {
  previous_cash_not_stored: "Деньги предыдущего клиента не убраны",
  cash_count_without_client: "Пересчёт денег в отсутствие клиента",
  cashier_left_during_service: "Уход во время обслуживания клиента",
  personal_items_in_cash_area: "Личные вещи в кассовой зоне",
  demo_violation: "Демонстрационное событие",
};

fetch("/health", { cache: "no-store" }).then(r => {
  if (!r.ok) throw new Error("health");
  return r.json();
}).then(info => {
  document.querySelector("#mode").textContent = info.demo
    ? "Демонстрационный режим: DummyPipeline создаёт фиктивное событие. Распознавание не выполняется."
    : "Анализ видео подключён. Система отмечает возможные нарушения; офицер проверяет кадры и применимость правила. Качество распознавания пока ограничено.";
}).catch(() => { document.querySelector("#mode").textContent = "Не удалось проверить подключение к сервису."; });

videoInput.addEventListener("change", () => {
  resultNode.hidden = true;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = videoInput.files[0] ? URL.createObjectURL(videoInput.files[0]) : null;
  document.querySelector("#preview").hidden = !videoUrl;
  document.querySelector("#player-note").textContent = "Нажмите таймкод события, чтобы перейти к нему в записи.";
  if (videoUrl) player.src = videoUrl;
  else player.removeAttribute("src");
});
player.addEventListener("error", () => {
  document.querySelector("#player-note").textContent = "Браузер не поддерживает кодек этого MOV. Анализ доступен; просмотрите исходник в VLC, а кадры — в карточках ниже.";
});
window.addEventListener("pagehide", () => { if (videoUrl) URL.revokeObjectURL(videoUrl); });

function seekButton(seconds) {
  const button = element("button", `${timecode(seconds)} · ${seconds.toFixed(2)} сек.`, "seek");
  button.type = "button";
  button.addEventListener("click", () => {
    if (videoUrl && !player.error) {
      player.currentTime = seconds;
      player.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  return button;
}

function showStatus(text, kind = "") {
  statusNode.textContent = text;
  statusNode.dataset.kind = kind;
}

function timecode(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${(seconds % 60).toFixed(2).padStart(5, "0")}`;
}

function element(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(path, {
      ...options,
      headers: { "X-API-Key": activeKey },
      signal: controller.signal,
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : `Ошибка HTTP ${response.status}`;
      throw new Error(detail);
    }
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

function renderResult(job) {
  resultNode.hidden = false;
  document.querySelector("#verdict").textContent = job.verdict === "violation"
    ? "Обнаружены признаки возможных нарушений"
    : "Нарушение не обнаружено";
  const list = document.querySelector("#violations");
  list.replaceChildren();
  for (const violation of job.violations) {
    const article = element("article", "", "violation");
    article.append(element("h3", titles[violation.type] || violation.type));
    article.append(seekButton(violation.timecode_sec));
    article.append(element("p", violation.description));
    article.append(element("p", `${violation.rule.document} · пункт ${violation.rule.clause}`, "rule"));
    const frames = element("div", "", "frames");
    for (const frame of violation.evidence_frames) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = `data:image/jpeg;base64,${frame.image_base64}`;
      image.alt = `Кадр на ${timecode(frame.timestamp_sec)}`;
      const expand = element("button", "", "seek");
      expand.type = "button";
      expand.setAttribute("aria-label", `Увеличить кадр на ${timecode(frame.timestamp_sec)}`);
      expand.append(image);
      expand.addEventListener("click", () => {
        document.querySelector("#full-frame").src = image.src;
        document.querySelector("#frame-dialog").showModal();
      });
      figure.append(expand, seekButton(frame.timestamp_sec));
      frames.append(figure);
    }
    article.append(frames);
    list.append(article);
  }
}

async function poll() {
  retryButton.hidden = true;
  submitButton.disabled = true;
  videoInput.disabled = true;
  try {
    while (activeJob) {
      const job = await api(`/api/v1/jobs/${encodeURIComponent(activeJob)}`);
      if (job.status === "done") {
        renderResult(job);
        showStatus(`Проверка завершена. Задание: ${job.job_id}`, "success");
        activeJob = null;
      } else if (job.status === "failed") {
        showStatus(`Проверка не завершена: ${job.error || "ошибка обработки"}. Задание: ${job.job_id}`, "error");
        activeJob = null;
      } else {
        showStatus(`Видео в очереди или обрабатывается. Задание: ${job.job_id}`);
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
  } catch (error) {
    showStatus(`Не удалось получить статус: ${error.message}. Задание: ${activeJob}`, "error");
    retryButton.hidden = false;
  } finally {
    submitButton.disabled = false;
    videoInput.disabled = Boolean(activeJob);
    if (!activeJob) activeKey = "";
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const file = document.querySelector("#video").files[0];
  if (!file || !file.name.toLowerCase().endsWith(".mov")) {
    showStatus("Выберите видео в формате .mov.", "error");
    return;
  }
  activeKey = document.querySelector("#api-key").value;
  activeJob = null;
  retryButton.hidden = true;
  resultNode.hidden = true;
  submitButton.disabled = true;
  videoInput.disabled = true;
  showStatus("Видео загружается…");
  try {
    const data = new FormData();
    data.append("file", file);
    const job = await api("/api/v1/analyze", { method: "POST", body: data });
    activeJob = job.job_id;
    await poll();
  } catch (error) {
    showStatus(`Ошибка загрузки: ${error.message}`, "error");
    activeKey = "";
    submitButton.disabled = false;
    videoInput.disabled = false;
  }
});

retryButton.addEventListener("click", () => {
  activeKey = document.querySelector("#api-key").value;
  void poll();
});
