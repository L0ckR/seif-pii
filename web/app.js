"use strict";

const $ = (id) => document.getElementById(id);
const source = $("source-text");
const examples = {
  client: "Карточка клиента\n\nКлиент: Иванов Иван Иванович\nДата рождения: 15.04.1990\nПаспорт: 4510 123456\nТелефон: +7 (900) 123-45-67\nEmail: ivan.petrov@example.org\nАдрес проживания: г. Москва, ул. Лесная, д. 12, кв. 34\n\nПросьба: объяснить условия вклада простыми словами.",
  context: "Подготовь справку для посетителей.\n\nАлександр Сергеевич Пушкин — русский поэт, автор романа «Евгений Онегин».\n\nПубличные реквизиты банка:\nПАО Сбербанк\nАдрес банка: 117312, г. Москва, ул. Вавилова, д. 19.\nБИК банка: 044525225\nИНН банка: 7707083893\n\nСравни описание банковского сервиса с литературной метафорой.",
  foreign: "Документы клиента / вымышленные данные\n\nКлиент: Петрова Анна Сергеевна\nДата рождения: 12.08.1985\nГражданство: Республика Беларусь\nИностранный паспорт: AB1234567\nЗагранпаспорт: 75 1234567\nВид на жительство: 123456789\nВодительское удостоверение: 77 12 345678\nEmail: anna.petrova@example.org\nТелефон: +1 (202) 555-0147\n\nЗадача: составить краткое описание обращения клиента."
};
const modeDescriptions = {
  token: "Именованные токены сохраняют структуру текста",
  mask: "Чувствительные фрагменты скрываются маской",
  synthetic: "Вымышленные значения вместо личных данных"
};
const reasonLabels = {
  context: "Явное обозначение поля в тексте",
  format: "Характерный формат значения",
  "russian-phone-format": "Формат российского телефонного номера",
  "international-phone-format": "Международный формат телефонного номера",
  checksum: "Контрольная сумма идентификатора",
  "explicit-card-context": "Номер указан в контексте банковской карты",
  "luhn-checksum": "Номер карты прошёл проверку контрольной суммы",
  "address-format": "Почтовый индекс в составе адреса",
  "patronymic-name": "Имя с отчеством и фамилией",
  "personal-record-context": "Имя в контексте клиентской записи",
  "surname-initials": "Фамилия и инициалы",
  "given-name-and-surname": "Сочетание имени и фамилии",
  "structured-address-context": "Структура личного адреса",
  "custom-rule": "Дополнительное правило политики системы"
};
let lastResult = null;
let busy = false;
let operationNumber = 0;
let typeLabels = {};
let noticeTimer;

function selectedMode() { return document.querySelector('input[name="mode"]:checked').value; }
function updateCount() {
  $("source-count").textContent = `${Array.from(source.value).length.toLocaleString("ru-RU")} символов`;
  $("protect-button").disabled = busy || !source.value.trim();
}
function notice(message, success = false) {
  clearTimeout(noticeTimer);
  const target = $("demo-notice");
  target.textContent = message;
  target.classList.toggle("success", success);
  target.hidden = !message;
}
function headers() {
  const result = { "Content-Type": "application/json" };
  const system = $("system-id").value.trim();
  const key = $("api-key").value.trim();
  if (system) result["X-System-ID"] = system;
  if (key) result["X-API-Key"] = key;
  return result;
}
function makeId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return `web-${Date.now()}-${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")}`;
}
async function api(path, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(path, { method: "POST", headers: headers(), body: JSON.stringify(body), signal: controller.signal });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error("Сервер вернул ответ в неожиданном формате. Проверьте доступность API."); }
    if (!response.ok) {
      const detail = data.detail ?? data.error;
      const hint = typeof detail === "string" ? detail : Array.isArray(detail) ? detail.map((item) => item.msg).filter(Boolean).join("; ") : detail && typeof detail.message === "string" ? detail.message : "Проверьте параметры запроса и настройки подключения.";
      throw new Error(`Ошибка ${response.status}: ${hint}`);
    }
    if (typeof data.result !== "string") throw new Error("В ответе API отсутствует текст результата.");
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Сервер не ответил за 30 секунд. Попробуйте ещё раз или уменьшите текст.");
    if (error instanceof TypeError) throw new Error("Не удалось связаться с API. Проверьте подключение и доступность сервера.");
    throw error;
  } finally { clearTimeout(timer); }
}
function setBusy(value, restoring = false) {
  busy = value;
  $("protect-button").disabled = value || !source.value.trim();
  $("restore-button").disabled = value || !lastResult || lastResult.restored;
  $("copy-button").disabled = value || !lastResult;
  $("clear-button").disabled = value;
  source.disabled = value;
  document.querySelectorAll('.example-tab, input[name="mode"], #system-id, #api-key').forEach((control) => { control.disabled = value; });
  $("protect-button").textContent = value && !restoring ? "Защищаем…" : "Защитить текст ↗";
  $("restore-button").textContent = value && restoring ? "Восстанавливаем…" : "↶ Восстановить";
  $("output-area").setAttribute("aria-busy", String(value));
}
function showOutput(value, restored = false) {
  const pre = document.createElement("pre");
  pre.className = `output-text${restored ? " restored" : ""}`;
  pre.textContent = value;
  $("output-area").replaceChildren(pre);
}
function showLoading(restoring) {
  const loader = document.createElement("div");
  loader.className = "loading-indicator";
  loader.textContent = restoring ? "Возвращаем исходные данные…" : "Анализируем контекст и защищаем данные…";
  $("output-area").replaceChildren(loader);
}
function emptyOutput() {
  const wrapper = document.createElement("div");
  wrapper.className = "output-empty";
  const symbol = document.createElement("div"); symbol.className = "empty-symbol"; symbol.textContent = "⌑"; symbol.setAttribute("aria-hidden", "true");
  const title = document.createElement("p"); title.textContent = "Здесь появится защищённый текст.";
  const subtitle = document.createElement("span"); subtitle.textContent = "Выберите режим и нажмите «Защитить»";
  wrapper.append(symbol, title, subtitle);
  $("output-area").replaceChildren(wrapper);
}
function labelFor(type) { return typeLabels[type] || typeLabels[String(type).toLowerCase()] || String(type || "Данные"); }
function reasonFor(reason) { return reasonLabels[reason] || reason || "Фрагмент соответствует признакам персональных данных."; }
function renderEntities(original, entities, payloadId) {
  // API offsets count Unicode code points, while JavaScript slice counts UTF-16 units.
  const codepoints = Array.from(original);
  const cleanEntities = Array.isArray(entities) ? entities.filter((entity) => entity && Number.isInteger(entity.start) && Number.isInteger(entity.end) && entity.start >= 0 && entity.end > entity.start && entity.end <= codepoints.length).sort((a, b) => a.start - b.start || b.end - a.end) : [];
  $("entity-total").textContent = String(cleanEntities.length);
  $("payload-meta").textContent = payloadId ? `ID: ${payloadId}` : "";
  $("results-detail").hidden = false;
  $("no-entities").hidden = cleanEntities.length > 0;
  const highlighted = document.createDocumentFragment();
  let offset = 0;
  for (const entity of cleanEntities) {
    if (entity.start < offset) continue;
    highlighted.append(document.createTextNode(codepoints.slice(offset, entity.start).join("")));
    const mark = document.createElement("mark");
    mark.className = "entity-mark";
    mark.textContent = codepoints.slice(entity.start, entity.end).join("");
    mark.title = `${labelFor(entity.type)}: ${reasonFor(entity.reason)}`;
    highlighted.append(mark);
    offset = entity.end;
  }
  highlighted.append(document.createTextNode(codepoints.slice(offset).join("")));
  $("highlighted-source").replaceChildren(highlighted);
  const list = document.createDocumentFragment();
  for (const entity of cleanEntities) {
    const li = document.createElement("li");
    const chip = document.createElement("span"); chip.className = "entity-chip"; chip.textContent = labelFor(entity.type);
    const explanation = document.createElement("span"); explanation.className = "entity-reason"; explanation.textContent = reasonFor(entity.reason);
    if (typeof entity.confidence === "number" && Number.isFinite(entity.confidence)) {
      const confidence = document.createElement("span"); confidence.className = "confidence"; confidence.textContent = ` · оценка детектора ${Math.round(entity.confidence * 100)}%`; explanation.append(confidence);
    }
    li.append(chip, explanation); list.append(li);
  }
  $("entities-list").replaceChildren(list);
  return cleanEntities.length;
}
function showTiming(data) {
  $("result-timing").textContent = typeof data.latency_ms === "number" && Number.isFinite(data.latency_ms) ? `${data.latency_ms.toLocaleString("ru-RU", { maximumFractionDigits: 2 })} ms` : "Время не передано";
}
async function protect() {
  if (busy || !source.value.trim()) return;
  const sequence = ++operationNumber;
  const original = source.value;
  const mode = selectedMode();
  const previous = lastResult;
  notice(""); setBusy(true); showLoading(false);
  $("result-status").textContent = "Выполняется запрос к API";
  $("result-timing").textContent = "— ms";
  try {
    const data = await api("/v1/mask", { payload: original, payload_id: makeId(), mode });
    if (sequence !== operationNumber) return;
    lastResult = { ...data, original, mode: data.mode || mode, restored: false };
    showOutput(data.result);
    const count = renderEntities(original, data.entities, data.payload_id);
    $("result-status").textContent = data.masking_enabled === false
      ? "Маскирование отключено политикой системы"
      : count ? `Защищено фрагментов: ${count} · ${Array.from(data.result).length.toLocaleString("ru-RU")} символов` : "Защищаемые сущности не найдены";
    showTiming(data);
  } catch (error) {
    if (sequence !== operationNumber) return;
    lastResult = previous;
    if (previous) showOutput(previous.result, previous.restored); else emptyOutput();
    $("result-status").textContent = "Запрос не выполнен";
    notice(error.message);
  } finally { if (sequence === operationNumber) setBusy(false); }
}
async function restore() {
  if (busy || !lastResult || lastResult.restored) return;
  const previous = lastResult;
  notice(""); setBusy(true, true); showLoading(true);
  try {
    const data = await api("/v1/unmask", { payload: previous.result, payload_id: previous.payload_id });
    lastResult = { ...previous, result: data.result, restored: true };
    showOutput(data.result, true);
    $("result-status").textContent = `Восстановлено · ${Array.from(data.result).length.toLocaleString("ru-RU")} символов`;
    showTiming(data);
    notice("Исходные данные восстановлены через API по идентификатору запроса.", true);
  } catch (error) { showOutput(previous.result); notice(error.message); }
  finally { setBusy(false); }
}
function resetResult() {
  lastResult = null;
  emptyOutput();
  $("results-detail").hidden = true;
  $("highlighted-source").replaceChildren();
  $("entities-list").replaceChildren();
  $("result-status").textContent = "Ожидает обработки";
  $("result-timing").textContent = "— ms";
  $("copy-button").disabled = true;
  $("restore-button").disabled = true;
  notice("");
}
function loadExample(button) {
  if (busy) return;
  document.querySelectorAll(".example-tab").forEach((tab) => {
    const active = tab === button;
    tab.classList.toggle("active", active); tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
  });
  source.value = examples[button.dataset.example];
  resetResult(); updateCount();
}

document.querySelectorAll(".example-tab").forEach((button, index, tabs) => {
  button.addEventListener("click", () => loadExample(button));
  button.addEventListener("keydown", (event) => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault(); tabs[next].focus(); loadExample(tabs[next]);
  });
});
source.addEventListener("input", () => {
  updateCount();
  document.querySelectorAll(".example-tab").forEach((tab) => { tab.classList.remove("active"); tab.setAttribute("aria-selected", "false"); });
  if (lastResult) $("result-status").textContent = "Результат предыдущего запроса · текст изменён";
});
source.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); protect(); } });
document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", () => { $("mode-help").textContent = modeDescriptions[selectedMode()]; }));
$("protect-button").addEventListener("click", protect);
$("restore-button").addEventListener("click", restore);
$("clear-button").addEventListener("click", () => { if (busy) return; source.value = ""; resetResult(); updateCount(); source.focus(); });
$("connection-toggle").addEventListener("click", () => {
  const expanded = $("connection-toggle").getAttribute("aria-expanded") === "true";
  $("connection-toggle").setAttribute("aria-expanded", String(!expanded));
  $("connection-settings").hidden = expanded;
});
$("copy-button").addEventListener("click", async () => {
  if (!lastResult) return;
  try { await navigator.clipboard.writeText(lastResult.result); notice("Текст скопирован.", true); noticeTimer = setTimeout(() => notice(""), 2200); }
  catch (_) { notice("Браузер не разрешил доступ к буферу обмена. Выделите результат и скопируйте его вручную."); }
});
async function initialize() {
  source.value = examples.client; updateCount();
  const outcomes = await Promise.allSettled([
    fetch("/health", { signal: AbortSignal.timeout(8000) }).then(async (response) => { if (!response.ok) throw new Error("Health failed"); return response.json(); }),
    fetch("/v1/types", { headers: headers(), signal: AbortSignal.timeout(8000) }).then(async (response) => { if (!response.ok) throw new Error("Types unavailable"); return response.json(); })
  ]);
  const health = outcomes[0];
  if (health.status === "fulfilled" && health.value.status === "ok") {
    $("health-status").classList.add("online");
    $("health-label").textContent = health.value.mode === "demo" ? "API доступен · demo" : "API доступен";
  } else { $("health-status").classList.add("offline"); $("health-label").textContent = "API недоступен"; }
  const types = outcomes[1];
  if (types.status === "fulfilled" && types.value.types && typeof types.value.types === "object") typeLabels = types.value.types;
}
initialize();
