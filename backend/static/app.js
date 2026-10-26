async function postForm(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  const data = await res.json();
  return { ok: res.ok, data };
}

function showResult(el, ok, data) {
  el.textContent = JSON.stringify(data, null, 2);
  el.className = "result " + (ok ? "success" : "error");
}

function fillSelect(select, items) {
  select.replaceChildren();
  items.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.id;
    opt.textContent = `${item.original_filename} (${item.validation_status})`;
    select.appendChild(opt);
  });
}

function refreshModelSelect(models) {
  fillSelect(document.getElementById("model-select"), models);
  fillSelect(document.getElementById("eval-model-select"),
    models.filter((m) => m.validation_status === "valid"));
}

async function loadModels() {
  const res = await fetch("/api/models");
  const models = await res.json();
  refreshModelSelect(models);
  return models;
}

function refreshDatasetSelect(datasets) {
  fillSelect(document.getElementById("dataset-select"), datasets);
  fillSelect(document.getElementById("eval-dataset-select"),
    datasets.filter((d) => d.validation_status === "valid"));
}

async function loadDatasets() {
  const res = await fetch("/api/datasets");
  const datasets = await res.json();
  refreshDatasetSelect(datasets);
  return datasets;
}

document.getElementById("model-upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const resultEl = document.getElementById("model-upload-result");
  const { ok, data } = await postForm("/api/models/upload", formData);
  showResult(resultEl, ok, data);
  if (ok) {
    form.reset();
    await loadModels();
  }
});

document.getElementById("model-validate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const modelId = document.getElementById("model-select").value;
  const resultEl = document.getElementById("model-validate-result");
  if (!modelId) {
    showResult(resultEl, false, { error: "Upload a model first." });
    return;
  }
  const res = await fetch(`/api/models/${modelId}/validate`, { method: "POST" });
  const data = await res.json();
  showResult(resultEl, res.ok, data);
  await loadModels();
});

document.getElementById("dataset-upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const resultEl = document.getElementById("dataset-upload-result");
  const { ok, data } = await postForm("/api/datasets/upload", formData);
  showResult(resultEl, ok, data);
  if (ok) {
    form.reset();
    await loadDatasets();
  }
});

document.getElementById("dataset-validate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const datasetId = document.getElementById("dataset-select").value;
  const resultEl = document.getElementById("dataset-validate-result");
  if (!datasetId) {
    showResult(resultEl, false, { error: "Upload a dataset first." });
    return;
  }
  const res = await fetch(`/api/datasets/${datasetId}/validate`, { method: "POST" });
  const data = await res.json();
  showResult(resultEl, res.ok, data);
  await loadDatasets();
});

// BL-06: use plain-language labels and explanations for the metrics returned by BL-05.
const METRIC_LABELS = {
  accuracy: {
    label: "Accuracy",
    help: "The proportion of predictions that were correct. Higher is better.",
  },
  precision: {
    label: "Weighted precision",
    help: "How often predictions for each class were correct, weighted by class size. Higher is better.",
  },
  recall: {
    label: "Weighted recall",
    help: "How many actual examples of each class were found, weighted by class size. Higher is better.",
  },
  f1_score: {
    label: "Weighted F1 score",
    help: "Balances precision and recall, weighted by class size. Higher is generally better.",
  },
  roc_auc: {
    label: "ROC-AUC",
    help: "Measures how well the classifier ranks positive examples above negative examples. Higher is better.",
  },
  mae: {
    label: "Mean absolute error (MAE)",
    help: "Average absolute prediction error, in the target's units. Lower is better.",
  },
  mse: {
    label: "Mean squared error (MSE)",
    help: "Average squared prediction error. Lower is better.",
  },
  rmse: {
    label: "Root mean squared error (RMSE)",
    help: "Square root of the mean squared error, in the target's units. Lower is better.",
  },
  r2: {
    label: "R² score",
    help: "Indicates how much target variance the model explains. Closer to 1 is generally better.",
  },
};

const CLASSIFICATION_METRICS = new Set(["accuracy", "precision", "recall", "f1_score", "roc_auc"]);

function formatMetric(value, key) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "Not available";
  }
  const number = Number(value);
  if (CLASSIFICATION_METRICS.has(key)) {
    return new Intl.NumberFormat(undefined, {
      style: "percent",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(number);
  }
  return number.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

function appendText(parent, tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.textContent = value;
  parent.appendChild(node);
  return node;
}

function renderEvaluationResult(container, record) {
  container.replaceChildren();
  container.className = "result success evaluation-result";

  const heading = document.createElement("div");
  heading.className = "evaluation-heading";
  appendText(heading, "h3", "", "Evaluation completed");
  appendText(heading, "p", "evaluation-context",
    `Model: ${record.model_filename || "Unknown model"} · Dataset: ${record.dataset_filename || "Unknown dataset"}`);
  appendText(heading, "p", "evaluation-context",
    `Task: ${record.task_type || "—"} · Rows evaluated: ${record.row_count ?? "—"} · Duration: ${record.duration_seconds ?? "—"} s`);
  container.appendChild(heading);

  const metrics = record.metrics || {};
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  Object.entries(metrics).forEach(([key, value]) => {
    const definition = METRIC_LABELS[key] || {
      label: key.replaceAll("_", " "),
      help: "Metric returned by the evaluation service.",
    };
    const card = document.createElement("article");
    card.className = "metric-card";
    appendText(card, "h4", "metric-label", definition.label);
    appendText(card, "p", "metric-value", formatMetric(value, key));
    appendText(card, "p", "metric-help", definition.help);
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
      appendText(card, "p", "metric-unavailable",
        key === "roc_auc"
          ? "ROC-AUC may be unavailable when the model does not support probability estimates or the data cannot produce a valid score. See the notes below."
          : "The evaluation did not return a numeric value for this metric.");
    }
    grid.appendChild(card);
  });
  if (!Object.keys(metrics).length) {
    appendText(grid, "p", "empty", "No metric values were returned for this evaluation.");
  }
  container.appendChild(grid);

  if (Array.isArray(record.notes) && record.notes.length) {
    const notes = document.createElement("div");
    notes.className = "evaluation-notes";
    appendText(notes, "h4", "", "Evaluation notes");
    const list = document.createElement("ul");
    record.notes.forEach((note) => appendText(list, "li", "", String(note)));
    notes.appendChild(list);
    container.appendChild(notes);
  }
  appendText(container, "p", "evaluation-timestamp",
    `Evaluated at: ${record.evaluated_at || "timestamp not provided"}`);
}

function renderEvaluationHistory(container, records) {
  container.replaceChildren();
  if (!Array.isArray(records) || records.length === 0) {
    appendText(container, "p", "empty", "No evaluations yet. Run an evaluation to see its metrics here.");
    return;
  }

  const table = document.createElement("table");
  table.className = "evaluation-history-table";
  const thead = document.createElement("thead");
  const header = document.createElement("tr");
  ["Evaluated at", "Model", "Dataset", "Task", "Metrics"].forEach((label) => {
    appendText(header, "th", "", label);
  });
  thead.appendChild(header);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  [...records].reverse().forEach((record) => {
    const row = document.createElement("tr");
    const time = record.evaluated_at ? new Date(record.evaluated_at) : null;
    const dateText = time && !Number.isNaN(time.getTime()) ? time.toLocaleString() : "—";
    appendText(row, "td", "", dateText);
    appendText(row, "td", "", record.model_filename || "Unknown model");
    appendText(row, "td", "", record.dataset_filename || "Unknown dataset");
    appendText(row, "td", "", record.task_type || "—");
    const summary = Object.entries(record.metrics || {})
      .map(([key, value]) => `${METRIC_LABELS[key]?.label || key}: ${formatMetric(value, key)}`)
      .join(" · ");
    appendText(row, "td", "history-metrics", summary || "No metrics");
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  container.appendChild(table);
}

async function loadEvaluationHistory() {
  const container = document.getElementById("evaluation-history");
  try {
    const response = await fetch("/api/evaluations");
    if (!response.ok) throw new Error("Evaluation history could not be loaded. Please refresh the page.");
    renderEvaluationHistory(container, await response.json());
  } catch (error) {
    container.replaceChildren();
    appendText(container, "p", "error-message", error.message || "Evaluation history could not be loaded.");
  }
}

document.getElementById("evaluate-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const resultEl = document.getElementById("evaluate-result");
  const payload = {
    model_id: document.getElementById("eval-model-select").value,
    dataset_id: document.getElementById("eval-dataset-select").value,
    task_type: document.getElementById("eval-task-select").value,
  };
  if (!payload.model_id || !payload.dataset_id) {
    showResult(resultEl, false, { error: "You need at least one validated model and one validated dataset." });
    return;
  }

  const button = e.submitter || e.target.querySelector('button[type="submit"]');
  if (button) {
    button.disabled = true;
    button.textContent = "Evaluating…";
  }
  resultEl.textContent = "Running evaluation…";
  resultEl.className = "result";
  try {
    const response = await fetch("/api/evaluations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      showResult(resultEl, false, data);
      return;
    }
    renderEvaluationResult(resultEl, data);
    await loadEvaluationHistory();
  } catch (error) {
    showResult(resultEl, false, { error: `Evaluation request failed: ${error.message}` });
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Run evaluation";
    }
  }
});

loadModels();
loadDatasets();
loadEvaluationHistory();
