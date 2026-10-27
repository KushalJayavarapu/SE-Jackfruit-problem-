// BL-06: plain-language metric labels and explanations.
const METRIC_LABELS = {
  accuracy: { label: "Accuracy", help: "The proportion of predictions that were correct. Higher is better." },
  precision: { label: "Weighted precision", help: "How often predictions for each class were correct, weighted by class size. Higher is better." },
  recall: { label: "Weighted recall", help: "How many actual examples of each class were found, weighted by class size. Higher is better." },
  f1_score: { label: "Weighted F1 score", help: "Balances precision and recall, weighted by class size. Higher is generally better." },
  roc_auc: { label: "ROC-AUC", help: "Measures how well the classifier ranks positive examples above negative examples. Higher is better." },
  mae: { label: "Mean absolute error (MAE)", help: "Average absolute prediction error, in the target's units. Lower is better." },
  mse: { label: "Mean squared error (MSE)", help: "Average squared prediction error. Lower is better." },
  rmse: { label: "Root mean squared error (RMSE)", help: "Square root of the mean squared error, in the target's units. Lower is better." },
  r2: { label: "R² score", help: "Indicates how much target variance the model explains. Closer to 1 is generally better." },
};
const CLASSIFICATION_METRICS = new Set(["accuracy", "precision", "recall", "f1_score", "roc_auc"]);

function appendText(parent, tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  parent.appendChild(node);
  return node;
}

function formatMetric(value, key) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "Not available";
  const number = Number(value);
  return CLASSIFICATION_METRICS.has(key)
    ? new Intl.NumberFormat(undefined, { style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(number)
    : number.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function errorMessage(data, fallback) {
  return data && typeof data.error === "string" ? data.error : fallback;
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_) {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  let data;
  try { data = await response.json(); }
  catch (_) { throw new Error("The server returned an unreadable response. Please try again."); }
  if (!response.ok) throw new Error(errorMessage(data, `The request failed (${response.status}). Please try again.`));
  return data;
}

function showMessage(element, message, kind = "") {
  element.textContent = message;
  element.className = `result${kind ? ` ${kind}` : ""}`;
}

async function withFormState(form, result, loadingText, action) {
  const button = form.querySelector('button[type="submit"]');
  const initialText = button.textContent;
  button.disabled = true;
  button.textContent = loadingText;
  showMessage(result, loadingText);
  try { await action(); }
  catch (error) { showMessage(result, error.message || "Something went wrong. Please try again.", "error"); }
  finally { button.disabled = false; button.textContent = initialText; }
}

function fillSelect(select, items, emptyMessage, filter = () => true) {
  const selected = select.value;
  select.replaceChildren();
  const available = items.filter(filter);
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = available.length ? "Choose an item" : emptyMessage;
  placeholder.disabled = true;
  placeholder.selected = true;
  select.appendChild(placeholder);
  available.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.original_filename} (${item.validation_status})`;
    option.selected = item.id === selected;
    select.appendChild(option);
  });
  select.disabled = available.length === 0;
}

function refreshModels(models) {
  fillSelect(document.getElementById("model-select"), models, "No models uploaded. Upload one above.");
  fillSelect(document.getElementById("eval-model-select"), models, "No validated models. Upload and validate a model first.", (m) => m.validation_status === "valid");
  const list = document.getElementById("model-list");
  list.replaceChildren();
  if (!models.length) appendText(list, "li", "empty", "No models uploaded yet. Add one above to get started.");
  models.forEach((m) => appendText(list, "li", "", `${m.original_filename} — ${m.validation_status}`));
}

function refreshDatasets(datasets) {
  fillSelect(document.getElementById("dataset-select"), datasets, "No datasets uploaded. Upload a CSV above.");
  fillSelect(document.getElementById("eval-dataset-select"), datasets, "No validated datasets. Upload and validate a dataset first.", (d) => d.validation_status === "valid");
  const list = document.getElementById("dataset-list");
  list.replaceChildren();
  if (!datasets.length) appendText(list, "li", "empty", "No datasets uploaded yet. Add a CSV above to get started.");
  datasets.forEach((d) => appendText(list, "li", "", `${d.original_filename} — ${d.row_count} rows, target: ${d.target_variable} — ${d.validation_status}`));
}

async function loadModels() { const models = await requestJson("/api/models"); if (!Array.isArray(models)) throw new Error("The server returned an unexpected model list."); refreshModels(models); return models; }
async function loadDatasets() { const datasets = await requestJson("/api/datasets"); if (!Array.isArray(datasets)) throw new Error("The server returned an unexpected dataset list."); refreshDatasets(datasets); return datasets; }

function renderEvaluationResult(container, record) {
  if (!record || typeof record !== "object") throw new Error("The server returned an unexpected evaluation result.");
  container.replaceChildren(); container.className = "result success evaluation-result";
  const heading = document.createElement("div"); heading.className = "evaluation-heading";
  appendText(heading, "h3", "", "Evaluation completed");
  appendText(heading, "p", "evaluation-context", `Model: ${record.model_filename || "Unknown model"} · Dataset: ${record.dataset_filename || "Unknown dataset"}`);
  appendText(heading, "p", "evaluation-context", `Task: ${record.task_type || "—"} · Rows evaluated: ${record.row_count ?? "—"} · Duration: ${record.duration_seconds ?? "—"} s`);
  container.appendChild(heading);
  const metrics = record.metrics && typeof record.metrics === "object" ? record.metrics : {};
  const grid = document.createElement("div"); grid.className = "metric-grid";
  Object.entries(metrics).forEach(([key, value]) => {
    const definition = METRIC_LABELS[key] || { label: key.replaceAll("_", " "), help: "Metric returned by the evaluation service." };
    const card = document.createElement("article"); card.className = "metric-card";
    appendText(card, "h4", "metric-label", definition.label); appendText(card, "p", "metric-value", formatMetric(value, key)); appendText(card, "p", "metric-help", definition.help);
    if (value === null || value === undefined || !Number.isFinite(Number(value))) appendText(card, "p", "metric-unavailable", key === "roc_auc" ? "ROC-AUC may be unavailable when the model does not support probability estimates or the data cannot produce a valid score. See the notes below." : "The evaluation did not return a numeric value for this metric.");
    grid.appendChild(card);
  });
  if (!Object.keys(metrics).length) appendText(grid, "p", "empty", "No metric values were returned for this evaluation.");
  container.appendChild(grid);
  if (Array.isArray(record.notes) && record.notes.length) {
    const notes = document.createElement("div"); notes.className = "evaluation-notes"; appendText(notes, "h4", "", "Evaluation notes");
    const list = document.createElement("ul"); record.notes.forEach((note) => appendText(list, "li", "", String(note))); notes.appendChild(list); container.appendChild(notes);
  }
  appendText(container, "p", "evaluation-timestamp", `Evaluated at: ${record.evaluated_at || "timestamp not provided"}`);
}

function renderEvaluationHistory(container, records) {
  container.replaceChildren();
  if (!Array.isArray(records) || records.length === 0) { appendText(container, "p", "empty", "No evaluations yet. Run an evaluation to see its metrics here."); return; }
  const table = document.createElement("table"); table.className = "evaluation-history-table";
  const thead = document.createElement("thead"); const header = document.createElement("tr");
  ["Evaluated at", "Model", "Dataset", "Task", "Metrics"].forEach((label) => { const th = appendText(header, "th", "", label); th.scope = "col"; });
  thead.appendChild(header); table.appendChild(thead);
  const tbody = document.createElement("tbody");
  [...records].reverse().forEach((record) => {
    const row = document.createElement("tr"); const time = record.evaluated_at ? new Date(record.evaluated_at) : null;
    appendText(row, "td", "", time && !Number.isNaN(time.getTime()) ? time.toLocaleString() : "—");
    appendText(row, "td", "", record.model_filename || "Unknown model"); appendText(row, "td", "", record.dataset_filename || "Unknown dataset"); appendText(row, "td", "", record.task_type || "—");
    const summary = Object.entries(record.metrics || {}).map(([key, value]) => `${METRIC_LABELS[key]?.label || key}: ${formatMetric(value, key)}`).join(" · ");
    appendText(row, "td", "history-metrics", summary || "No metrics"); tbody.appendChild(row);
  });
  table.appendChild(tbody); container.appendChild(table);
}

async function loadEvaluationHistory() {
  const container = document.getElementById("evaluation-history");
  try { const records = await requestJson("/api/evaluations"); renderEvaluationHistory(container, records); }
  catch (error) { container.replaceChildren(); appendText(container, "p", "error-message", error.message); }
}

function wireForm(id, resultId, loadingText, action) {
  const form = document.getElementById(id); const result = document.getElementById(resultId);
  form.addEventListener("submit", (event) => { event.preventDefault(); if (form.dataset.busy === "true") return; form.dataset.busy = "true"; withFormState(form, result, loadingText, action).finally(() => { form.dataset.busy = "false"; }); });
}

wireForm("model-upload-form", "model-upload-result", "Uploading model…", async () => {
  const form = document.getElementById("model-upload-form");
  const data = await requestJson("/api/models/upload", { method: "POST", body: new FormData(form) });
  showMessage(document.getElementById("model-upload-result"), `Model uploaded: ${data.original_filename || "file received"}. Validate it before evaluation.`, "success"); form.reset(); await loadModels();
});
wireForm("model-validate-form", "model-validate-result", "Validating model…", async () => {
  const id = document.getElementById("model-select").value;
  if (!id) throw new Error("Upload a model first, then select it here.");
  const data = await requestJson(`/api/models/${encodeURIComponent(id)}/validate`, { method: "POST" });
  showMessage(document.getElementById("model-validate-result"), data.validation_status === "valid" ? "Model validated and ready for evaluation." : `Model validation failed: ${data.validation_message || "check the model file and try again."}`, data.validation_status === "valid" ? "success" : "error"); await loadModels();
});
wireForm("dataset-upload-form", "dataset-upload-result", "Uploading dataset…", async () => {
  const form = document.getElementById("dataset-upload-form");
  const data = await requestJson("/api/datasets/upload", { method: "POST", body: new FormData(form) });
  showMessage(document.getElementById("dataset-upload-result"), `Dataset uploaded: ${data.original_filename || "CSV received"}. Validate it before evaluation.`, "success"); form.reset(); await loadDatasets();
});
wireForm("dataset-validate-form", "dataset-validate-result", "Validating dataset…", async () => {
  const id = document.getElementById("dataset-select").value;
  if (!id) throw new Error("Upload a dataset first, then select it here.");
  const data = await requestJson(`/api/datasets/${encodeURIComponent(id)}/validate`, { method: "POST" });
  showMessage(document.getElementById("dataset-validate-result"), data.validation_status === "valid" ? "Dataset validated and ready for evaluation." : `Dataset validation failed: ${data.validation_message || "check the target column and dataset format."}`, data.validation_status === "valid" ? "success" : "error"); await loadDatasets();
});
wireForm("evaluate-form", "evaluate-result", "Running evaluation…", async () => {
  const payload = { model_id: document.getElementById("eval-model-select").value, dataset_id: document.getElementById("eval-dataset-select").value, task_type: document.getElementById("eval-task-select").value };
  if (!payload.model_id || !payload.dataset_id) throw new Error("You need a validated model and dataset. Upload and validate both first.");
  const data = await requestJson("/api/evaluations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  renderEvaluationResult(document.getElementById("evaluate-result"), data); await loadEvaluationHistory();
});

(async function initialize() {
  for (const [loader, resultId] of [[loadModels, "model-upload-result"], [loadDatasets, "dataset-upload-result"]]) {
    try { await loader(); } catch (error) { showMessage(document.getElementById(resultId), error.message, "error"); }
  }
  await loadEvaluationHistory();
})();
