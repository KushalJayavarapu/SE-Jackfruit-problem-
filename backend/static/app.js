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
  select.innerHTML = "";
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
  const res = await fetch("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  showResult(resultEl, res.ok, data);
});

loadModels();
loadDatasets();
