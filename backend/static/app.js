async function postForm(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  const data = await res.json();
  return { ok: res.ok, data };
}

function showResult(el, ok, data) {
  el.textContent = JSON.stringify(data, null, 2);
  el.className = "result " + (ok ? "success" : "error");
}

function refreshModelSelect(models) {
  const select = document.getElementById("model-select");
  select.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.original_filename} (${m.validation_status})`;
    select.appendChild(opt);
  });
}

async function loadModels() {
  const res = await fetch("/api/models");
  const models = await res.json();
  refreshModelSelect(models);
  return models;
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

loadModels();
