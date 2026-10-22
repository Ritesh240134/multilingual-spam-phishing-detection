const MAX_CHARS = 200_000;
const API_BASE = (window.SIGNALCHECK_CONFIG?.apiBaseUrl || "http://127.0.0.1:8000").replace(/\/$/, "");
const VALID_LABELS = new Set(["legitimate", "spam", "phishing"]);
let currentModelStatus = "unavailable";

class InputValidationError extends Error {}

export function validateEmail({ subject, body, sender, urls }) {
  if (!subject.trim() && !body.trim()) return "Enter a subject or email body to continue.";
  if (body.length > MAX_CHARS) return "The email body exceeds the 200,000 character limit.";
  const urlList = urls.split(/\r?\n/).map((url) => url.trim()).filter(Boolean);
  if (urlList.length > 100) return "Add no more than 100 URLs.";
  if (urlList.some((url) => url.length > MAX_CHARS)) return "One or more URLs are too long.";
  const size = subject.length + body.length + sender.length + urlList.reduce((sum, url) => sum + url.length, 0);
  if (size > MAX_CHARS) return "The combined input exceeds the 200,000 character limit.";
  return null;
}

export async function analyzeEmail(fields, fetchImpl = fetch, baseUrl = API_BASE) {
  const validationError = validateEmail(fields);
  if (validationError) throw new InputValidationError(validationError);
  const urls = fields.urls.split(/\r?\n/).map((url) => url.trim()).filter(Boolean);
  const payload = {
    subject: fields.subject || null,
    sender: fields.sender || null,
    body: fields.body || null,
    urls: urls.length ? urls : null,
  };
  const response = await fetchImpl(`${baseUrl.replace(/\/$/, "")}/api/v1/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (response.status === 422) {
    const detail = await response.json().catch(() => null);
    throw new TypeError(detail?.detail?.[0]?.msg || "The API rejected the input. Check the fields and try again.");
  }
  if (!response.ok) throw new Error(`The analysis service returned HTTP ${response.status}.`);
  return response.json();
}

function addText(parent, tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = value;
  parent.append(element);
  return element;
}

function setConnection(id, textId, label, state) {
  const dot = document.querySelector(`#${id}`);
  const text = document.querySelector(`#${textId}`);
  if (!dot || !text) return;
  dot.className = `status-dot ${state}`;
  text.textContent = label;
}

function updateModelStatus(modelStatus) {
  const states = {
    available: ["Available", "ready"],
    unavailable: ["Unavailable", "unavailable"],
    error: ["Error", "error"],
  };
  const [label, state] = states[modelStatus] || ["Not checked", "pending"];
  currentModelStatus = modelStatus;
  setConnection("model-dot", "model-status", label, state);
  const overview = document.querySelector("#overview-model");
  if (overview) overview.textContent = label;
}

async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { method: "GET", headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error("API health check failed");
    const health = await response.json();
    if (health.status !== "ok") throw new Error("API is not ready");
    setConnection("api-dot", "api-status", "Online", "online");
  } catch {
    setConnection("api-dot", "api-status", "Offline", "offline");
  }
}

function fieldsFrom(form) {
  return {
    subject: form.elements.subject.value,
    sender: form.elements.sender.value,
    body: form.elements.body.value,
    urls: form.elements.urls.value,
  };
}

function setResultStatus(text, kind = "neutral") {
  const status = document.querySelector("#result-status");
  status.className = `result-badge ${kind}`;
  const dot = document.createElement("span");
  dot.className = `status-dot ${kind === "success" ? "ready" : kind === "danger" ? "error" : kind === "warning" ? "unavailable" : "pending"}`;
  dot.setAttribute("aria-hidden", "true");
  status.replaceChildren(dot, document.createTextNode(text));
}

function renderMessage(container, message, kind = "warning") {
  container.replaceChildren();
  addText(container, "p", `result-message ${kind}`, message);
}

function renderResult(data) {
  const empty = document.querySelector("#empty-state");
  const notice = document.querySelector("#model-notice");
  const details = document.querySelector("#result-details");
  empty.hidden = true;
  notice.hidden = true;
  details.hidden = false;
  details.replaceChildren();
  updateModelStatus(data.model_status);

  if (data.model_status === "unavailable" || data.inference_status === "unavailable") {
    setResultStatus("Model unavailable", "warning");
    renderMessage(details, data.explanation || "No trained local classifier is configured; no prediction was made.");
  } else if (data.model_status === "error" || data.inference_status === "error") {
    setResultStatus("Analysis error", "danger");
    renderMessage(details, data.explanation || "The API could not complete this analysis.", "error");
  } else if (data.model_status === "available" && data.inference_status === "completed" && VALID_LABELS.has(data.label)) {
    setResultStatus("Analysis complete", data.label === "legitimate" ? "success" : "warning");
    const classification = document.createElement("div");
    classification.className = "result-classification";
    const marker = document.createElement("span");
    marker.className = `class-indicator ${data.label}`;
    marker.setAttribute("aria-hidden", "true");
    classification.append(marker);
    addText(classification, "h3", "", data.label);
    details.append(classification);
    if (typeof data.explanation === "string" && data.explanation.trim()) {
      addText(details, "p", "result-explanation", data.explanation);
    }
  } else {
    setResultStatus("No prediction", "warning");
    renderMessage(details, "The API did not return a completed model prediction.");
  }

  const facts = document.createElement("div");
  facts.className = "result-facts";
  if (typeof data.risk_score === "number" && Number.isFinite(data.risk_score)) {
    const fact = document.createElement("div");
    fact.className = "result-fact";
    addText(fact, "span", "", "Risk score");
    addText(fact, "strong", "", `${(data.risk_score * 100).toFixed(1)}%`);
    facts.append(fact);
  }
  if (typeof data.detected_language === "string" && data.detected_language.trim()) {
    const fact = document.createElement("div");
    fact.className = "result-fact";
    addText(fact, "span", "", "Detected language");
    addText(fact, "strong", "", data.detected_language);
    facts.append(fact);
  }
  if (facts.childElementCount) details.append(facts);

  const indicators = data.security_indicators;
  if (!indicators || typeof indicators !== "object") return;
  const entries = [
    ["URLs found", indicators.url_count],
    ["Suspicious URLs", indicators.suspicious_url_count],
    ["IP-based URLs", indicators.ip_address_url_count],
    ["Shortened URLs", indicators.shortener_url_count],
  ].filter(([, value]) => Number.isInteger(value) && value >= 0);
  if (entries.length) {
    addText(details, "h3", "indicator-heading", "Security indicators");
    const tags = document.createElement("div");
    tags.className = "indicator-list";
    for (const [label, value] of entries) {
      const tag = addText(tags, "span", "indicator-tag", `${label}: ${value}`);
      if (label !== "URLs found" && value > 0) tag.classList.add("alert");
    }
    details.append(tags);
  }
  if (Array.isArray(indicators.suspicious_urls) && indicators.suspicious_urls.length) {
    addText(details, "h3", "suspicious-title", "Suspicious URL details");
    const list = document.createElement("ul");
    list.className = "suspicious-list";
    for (const url of indicators.suspicious_urls) {
      if (typeof url === "string") addText(list, "li", "", url);
    }
    if (list.childElementCount) details.append(list);
  }
}

const form = document.querySelector("#analyze-form");
const appShell = document.querySelector(".app-shell");
const sidebarToggle = document.querySelector("#sidebar-toggle");

function setNavigationCollapsed(collapsed) {
  if (!appShell || !sidebarToggle) return;
  appShell.classList.toggle("nav-collapsed", collapsed);
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  const action = collapsed ? "Expand" : "Collapse";
  sidebarToggle.setAttribute("aria-label", action + " navigation");
  sidebarToggle.title = action + " navigation";
}

function setActiveNavigation(target) {
  document.querySelectorAll(".nav-link").forEach((item) => {
    const isActive = item.getAttribute("href") === target;
    item.classList.toggle("active", isActive);
    if (isActive) item.setAttribute("aria-current", "page");
    else item.removeAttribute("aria-current");
  });
}

sidebarToggle?.addEventListener("click", () => {
  setNavigationCollapsed(!appShell.classList.contains("nav-collapsed"));
});

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", () => {
    setActiveNavigation(link.getAttribute("href"));
    if (window.matchMedia("(max-width: 760px)").matches) setNavigationCollapsed(true);
  });
});

document.querySelector(".brand")?.addEventListener("click", () => setActiveNavigation("#overview"));
document.querySelector(".primary-link")?.addEventListener("click", () => setActiveNavigation("#analyzer"));

if (form) {
  const bodyField = form.elements.body;
  const button = document.querySelector("#submit-button");
  const resetButton = document.querySelector("#reset-button");
  const submitLabel = document.querySelector("#submit-label");
  const submitIcon = document.querySelector("#submit-icon");
  const errorBox = document.querySelector("#form-error");
  const noticeBox = document.querySelector("#form-notice");
  const details = document.querySelector("#result-details");
  const empty = document.querySelector("#empty-state");
  const modelNotice = document.querySelector("#model-notice");

  checkApiHealth();
  bodyField.addEventListener("input", () => {
    document.querySelector("#char-count").textContent = `${bodyField.value.length.toLocaleString()} / 200,000`;
  });

  form.addEventListener("reset", () => {
    requestAnimationFrame(() => {
      document.querySelector("#char-count").textContent = "0 / 200,000";
      errorBox.hidden = true;
      noticeBox.hidden = true;
      details.replaceChildren();
      details.hidden = true;
      empty.hidden = false;
      modelNotice.hidden = currentModelStatus !== "unavailable";
      setResultStatus("Awaiting input");
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    noticeBox.hidden = true;
    const fields = fieldsFrom(form);
    const validationError = validateEmail(fields);
    if (validationError) {
      errorBox.textContent = validationError;
      errorBox.hidden = false;
      form.elements.subject.focus();
      return;
    }

    button.disabled = true;
    resetButton.disabled = true;
    button.setAttribute("aria-busy", "true");
    submitLabel.textContent = "Analyzing";
    const spinner = document.createElement("span");
    spinner.className = "loading-spinner";
    spinner.id = "loading-spinner";
    spinner.setAttribute("aria-hidden", "true");
    submitIcon.replaceWith(spinner);
    setResultStatus("Analyzing", "neutral");
    empty.hidden = true;
    modelNotice.hidden = true;
    details.hidden = false;
    renderMessage(details, "Your request is being processed by the configured analysis API.", "");

    try {
      const data = await analyzeEmail(fields);
      renderResult(data);
    } catch (error) {
      setResultStatus("Request failed", "danger");
      renderMessage(details, error instanceof TypeError ? error.message : "Could not reach the analysis API. Check that the backend is running and try again.", "error");
      errorBox.textContent = error instanceof TypeError ? error.message : "The analysis request failed. Your form contents are still available; check the API connection and retry.";
      errorBox.hidden = false;
    } finally {
      button.disabled = false;
      resetButton.disabled = false;
      button.removeAttribute("aria-busy");
      submitLabel.textContent = "Analyze email";
      const spinner = document.querySelector("#loading-spinner");
      if (spinner) spinner.replaceWith(submitIcon);
    }
  });
}
