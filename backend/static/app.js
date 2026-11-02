// Model Evaluator page scripts. One file, one init function per page.
// All text from the server is inserted with textContent, never innerHTML.

const csrfToken = () => document.querySelector('meta[name="csrf-token"]').content;
const $ = (selector, root = document) => root.querySelector(selector);

// ---------------------------------------------------------------- helpers

function el(tag, options = {}, ...children) {
  const node = document.createElement(tag);
  const { className, text, attrs, on, value } = options;
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  if (value !== undefined) node.value = value;
  if (attrs) for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  if (on) for (const [k, v] of Object.entries(on)) node.addEventListener(k, v);
  for (const child of children) if (child) node.append(child);
  return node;
}

async function api(url, options = {}) {
  const headers = { "X-CSRF-Token": csrfToken(), ...(options.headers || {}) };
  let response;
  try {
    response = await fetch(url, { ...options, headers, credentials: "same-origin" });
  } catch (_) {
    throw new Error("Could not reach the server. Check your connection and try again.");
  }
  if (response.status === 401) { window.location.href = "/login"; throw new Error("Please log in to continue."); }
  let data = null;
  try { data = await response.json(); } catch (_) { /* not JSON */ }
  if (!response.ok) {
    throw new Error(data && typeof data.error === "string" ? data.error : `The request failed (${response.status}). Please try again.`);
  }
  return data;
}

const postJson = (url, body) => api(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

function showResult(node, message, kind = "") {
  node.textContent = message;
  node.className = `result${kind ? " " + kind : ""}`;
}

let toastTimer;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 3200);
}

function confirmDialog(title, body, okLabel = "Confirm") {
  const dialog = $("#confirm-dialog");
  $("#confirm-title").textContent = title;
  $("#confirm-body").textContent = body;
  $("#confirm-ok").textContent = okLabel;
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
    dialog.showModal();
  });
}

async function busy(button, label, action) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try { return await action(); }
  finally { button.disabled = false; button.textContent = original; }
}

function fmtBytes(n) {
  if (n >= 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  if (n >= 1024) return `${Math.round(n / 1024)} KB`;
  return `${n} bytes`;
}

function fmtDate(iso) {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function pill(status) {
  const labels = { valid: "Valid", invalid: "Invalid", not_validated: "Not validated", active: "Active", suspended: "Suspended" };
  return el("span", { className: `pill pill-${status}`, text: labels[status] || status });
}

function headlineMetric(e) {
  const m = e.metrics || {};
  return e.task_type === "classification"
    ? `Accuracy ${Number(m.accuracy).toFixed(3)}`
    : `R² ${Number(m.r2).toFixed(3)}`;
}

function wireFilePicker(input) {
  const label = input.parentElement.querySelector(".filepick-label");
  input.addEventListener("change", () => {
    const file = input.files[0];
    label.textContent = file ? file.name : label.dataset.empty;
    input.parentElement.classList.toggle("has-file", Boolean(file));
  });
  return () => { label.textContent = label.dataset.empty; input.parentElement.classList.remove("has-file"); };
}

function downloadReport(ids) {
  window.location.href = `/api/reports?ids=${encodeURIComponent(ids.join(","))}`;
}

// ---------------------------------------------------------------- models page

function renderModels(list, models) {
  list.replaceChildren();
  if (!models.length) {
    list.append(el("p", { className: "empty", text: "No models yet. Upload a .pkl or .joblib file above." }));
    return;
  }
  for (const m of [...models].reverse()) list.append(modelCard(m, list));
}

function modelCard(m, list) {
  const card = el("article", { className: "record", attrs: { "data-id": m.id } });
  const actions = el("div", { className: "record-actions" });
  const validate = el("button", {
    className: m.validation_status === "valid" ? "secondary small" : "small",
    text: m.validation_status === "valid" ? "Validate again" : "Validate",
    on: { click: () => busy(validate, "Validating…", async () => {
      try {
        const updated = await api(`/api/models/${m.id}/validate`, { method: "POST" });
        card.replaceWith(modelCard(updated, list));
        toast(updated.validation_status === "valid" ? "Model is valid." : "Model is not valid. See the message on the card.");
      } catch (error) { toast(error.message); }
    }) },
  });
  const remove = el("button", {
    className: "danger small", text: "Delete",
    on: { click: async () => {
      if (!await confirmDialog("Delete this model?", `${m.original_filename} will be removed. Past evaluations that used it stay in your history.`, "Delete model")) return;
      try { await api(`/api/models/${m.id}`, { method: "DELETE" }); card.remove(); toast("Model deleted."); if (!list.children.length) renderModels(list, []); }
      catch (error) { toast(error.message); }
    } },
  });
  actions.append(validate, remove);

  card.append(el("div", { className: "record-top" },
    el("div", {},
      el("p", { className: "record-name" }, pill(m.validation_status), document.createTextNode(m.original_filename)),
      el("p", { className: "record-meta", text: `${fmtBytes(m.size_bytes)} · uploaded ${fmtDate(m.uploaded_at)}` })),
    actions));

  if (m.validation_message) {
    card.append(el("p", { className: `record-msg${m.validation_status === "invalid" ? " bad" : ""}`, text: m.validation_message }));
  }
  const inputs = m.required_inputs;
  if (inputs) {
    const facts = el("div", { className: "facts-inline" }, el("span", { text: `Type: ${inputs.model_type}` }),
      el("span", { text: inputs.is_classifier ? "Predicts categories (classification)" : "Predicts numbers (regression)" }),
      el("span", { text: inputs.has_predict_proba ? "Gives probabilities (ROC-AUC available)" : "No probabilities (no ROC-AUC)" }),
      el("span", { text: `${inputs.n_features_in ?? "Unknown number of"} input columns` }));
    card.append(facts);
    if (inputs.feature_names_in) {
      card.append(el("details", { className: "more" }, el("summary", { text: "Columns this model expects" }),
        el("div", { className: "cols" }, ...inputs.feature_names_in.map((n) => el("span", { className: "col-chip", text: n })))));
    }
  }
  if (m.compatibility && m.compatibility.length) card.append(compatList("Fit with your datasets", m.compatibility, "dataset_filename"));
  return card;
}

function compatList(title, items, nameKey) {
  return el("div", {}, el("p", { className: "record-meta", text: title }),
    el("ul", { className: "compat" }, ...items.map((c) =>
      el("li", { className: c.compatible ? "ok" : "no", text: c.compatible ? `${c[nameKey]}: compatible` : `${c[nameKey]}: ${c.reason}` }))));
}

function initModels() {
  const form = $("#model-upload-form");
  const result = $("#model-upload-result");
  const list = $("#model-list");
  const reset = wireFilePicker($("#model-file"));
  const load = async () => { try { renderModels(list, await api("/api/models")); } catch (e) { list.replaceChildren(el("p", { className: "empty", text: e.message })); } };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type=submit]");
    await busy(button, "Uploading…", async () => {
      showResult(result, "Uploading…");
      try {
        const record = await api("/api/models/upload", { method: "POST", body: new FormData(form) });
        showResult(result, `Uploaded ${record.original_filename}. Press Validate on its card to check it.`, "success");
        form.reset(); reset(); await load();
      } catch (error) { showResult(result, error.message, "error"); }
    });
  });
  load();
}

// ---------------------------------------------------------------- datasets page

function parseCsvHeader(text) {
  const line = text.split(/\r?\n/, 1)[0].replace(/^﻿/, "");
  const names = []; let current = ""; let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { current += '"'; i++; }
      else if (ch === '"') quoted = false;
      else current += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { names.push(current); current = ""; }
    else current += ch;
  }
  names.push(current);
  return names.map((n) => n.trim()).filter(Boolean);
}

function renderDatasets(list, datasets) {
  list.replaceChildren();
  if (!datasets.length) {
    list.append(el("p", { className: "empty", text: "No datasets yet. Upload a CSV file above." }));
    return;
  }
  for (const d of [...datasets].reverse()) list.append(datasetCard(d, list));
}

function datasetCard(d, list) {
  const card = el("article", { className: "record", attrs: { "data-id": d.id } });
  const validate = el("button", {
    className: d.validation_status === "valid" ? "secondary small" : "small",
    text: d.validation_status === "valid" ? "Validate again" : "Validate",
    on: { click: () => busy(validate, "Validating…", async () => {
      try {
        const updated = await api(`/api/datasets/${d.id}/validate`, { method: "POST" });
        card.replaceWith(datasetCard(updated, list));
        toast(updated.validation_status === "valid" ? "Dataset is valid." : "Dataset is not valid. See the message on the card.");
      } catch (error) { toast(error.message); }
    }) },
  });
  const remove = el("button", {
    className: "danger small", text: "Delete",
    on: { click: async () => {
      if (!await confirmDialog("Delete this dataset?", `${d.original_filename} will be removed. Past evaluations that used it stay in your history.`, "Delete dataset")) return;
      try { await api(`/api/datasets/${d.id}`, { method: "DELETE" }); card.remove(); toast("Dataset deleted."); if (!list.children.length) renderDatasets(list, []); }
      catch (error) { toast(error.message); }
    } },
  });
  card.append(el("div", { className: "record-top" },
    el("div", {},
      el("p", { className: "record-name" }, pill(d.validation_status), document.createTextNode(d.original_filename)),
      el("p", { className: "record-meta", text: `${d.row_count.toLocaleString()} rows · target column "${d.target_variable}" · uploaded ${fmtDate(d.uploaded_at)}` })),
    el("div", { className: "record-actions" }, validate, remove)));
  if (d.validation_message) {
    card.append(el("p", { className: `record-msg${d.validation_status === "invalid" ? " bad" : ""}`, text: d.validation_message }));
  }
  if (d.schema_summary) {
    const s = d.schema_summary;
    const missing = s.columns.reduce((sum, c) => sum + c.missing_count, 0);
    card.append(el("div", { className: "facts-inline" },
      el("span", { text: `${s.feature_columns.length} feature columns` }),
      el("span", { text: `${missing} missing values` }),
      el("span", { text: `${s.duplicate_rows} duplicate rows` })));
    card.append(el("details", { className: "more" }, el("summary", { text: "Columns" }),
      el("div", { className: "cols" }, ...s.columns.map((c) =>
        el("span", { className: `col-chip${c.is_target ? " target" : ""}`, text: `${c.name} · ${c.dtype}${c.missing_count ? ` · ${c.missing_count} missing` : ""}` })))));
  }
  if (d.compatibility && d.compatibility.length) card.append(compatList("Fit with your models", d.compatibility, "model_filename"));
  return card;
}

function initDatasets() {
  const form = $("#dataset-upload-form");
  const result = $("#dataset-upload-result");
  const list = $("#dataset-list");
  const fileInput = $("#dataset-file");
  const target = $("#target-variable");
  const resetPicker = wireFilePicker(fileInput);
  const load = async () => { try { renderDatasets(list, await api("/api/datasets")); } catch (e) { list.replaceChildren(el("p", { className: "empty", text: e.message })); } };

  const resetTarget = (text) => { target.replaceChildren(el("option", { value: "", text })); target.disabled = true; };
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files[0];
    if (!file) return resetTarget("Choose a file first");
    try {
      const names = parseCsvHeader(await file.slice(0, 65536).text());
      if (!names.length) throw new Error("empty");
      target.replaceChildren(el("option", { value: "", text: "Choose the column to predict" }),
        ...names.map((n) => el("option", { value: n, text: n })));
      target.disabled = false;
      const guess = names.find((n) => /^(target|label|class|y|outcome)$/i.test(n));
      if (guess) target.value = guess;
    } catch (_) {
      resetTarget("Could not read column names");
      showResult(result, "Could not read the column names from this file. Check that it is a CSV with a header row.", "error");
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type=submit]");
    await busy(button, "Uploading…", async () => {
      showResult(result, "Uploading…");
      try {
        const record = await api("/api/datasets/upload", { method: "POST", body: new FormData(form) });
        showResult(result, `Uploaded ${record.original_filename}. Press Validate on its card to check it.`, "success");
        form.reset(); resetPicker(); resetTarget("Choose a file first"); await load();
      } catch (error) { showResult(result, error.message, "error"); }
    });
  });
  load();
}

// ---------------------------------------------------------------- evaluate page

async function initEvaluate() {
  const form = $("#evaluate-form");
  const empty = $("#evaluate-empty");
  const modelSelect = $("#eval-model");
  const datasetSelect = $("#eval-dataset");
  const pairCheck = $("#pair-check");
  const result = $("#evaluate-result");

  let models = [], datasets = [];
  try {
    [models, datasets] = await Promise.all([api("/api/models"), api("/api/datasets")]);
  } catch (error) { empty.hidden = false; empty.textContent = error.message; return; }
  models = models.filter((m) => m.validation_status === "valid");
  datasets = datasets.filter((d) => d.validation_status === "valid");

  if (!models.length || !datasets.length) {
    empty.hidden = false;
    const missing = [!models.length && "a validated model", !datasets.length && "a validated dataset"].filter(Boolean).join(" and ");
    empty.replaceChildren(document.createTextNode(`You need ${missing} first. `),
      !models.length ? el("a", { text: "Go to Models", attrs: { href: "/models" } }) : el("a", { text: "Go to Datasets", attrs: { href: "/datasets" } }));
    return;
  }
  form.hidden = false;
  modelSelect.replaceChildren(...models.map((m) => el("option", { value: m.id, text: m.original_filename })));
  datasetSelect.replaceChildren(...datasets.map((d) => el("option", { value: d.id, text: `${d.original_filename} (target: ${d.target_variable})` })));

  const radios = [...form.querySelectorAll('input[name="task_type"]')];
  const check = () => {
    const m = models.find((x) => x.id === modelSelect.value);
    const d = datasets.find((x) => x.id === datasetSelect.value);
    const inputs = m.required_inputs || {};
    const cols = d.columns.filter((c) => c !== d.target_variable);
    let reason = null;
    if (inputs.feature_names_in) {
      const missing = inputs.feature_names_in.filter((c) => !cols.includes(c));
      if (missing.length) reason = `This model expects columns the dataset does not have: ${missing.join(", ")}.`;
    } else if (inputs.n_features_in != null && inputs.n_features_in !== cols.length) {
      reason = `This model expects ${inputs.n_features_in} columns but the dataset has ${cols.length} besides the target.`;
    }
    pairCheck.className = `pair-check ${reason ? "bad" : "ok"}`;
    pairCheck.textContent = reason || "This model and dataset fit together.";
    // Suggest the task type that matches the model.
    if (!radios.some((r) => r.checked) && inputs.is_classifier !== undefined) {
      radios.find((r) => r.value === (inputs.is_classifier ? "classification" : "regression")).checked = true;
    }
  };
  modelSelect.addEventListener("change", check);
  datasetSelect.addEventListener("change", check);
  check();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const task = form.querySelector('input[name="task_type"]:checked');
    if (!task) return showResult(result, "Choose a task type.", "error");
    const button = form.querySelector("button[type=submit]");
    await busy(button, "Evaluating…", async () => {
      showResult(result, "Running the model on your dataset…");
      try {
        const record = await postJson("/api/evaluations", {
          model_id: modelSelect.value, dataset_id: datasetSelect.value, task_type: task.value });
        window.location.href = `/evaluations/${record.id}`;
      } catch (error) { showResult(result, error.message, "error"); }
    });
  });
}

// ---------------------------------------------------------------- history page

async function initHistory() {
  const container = $("#evaluation-history");
  const status = $("#selection-status");
  const compareButton = $("#compare-button");
  const reportButton = $("#report-button");
  const warning = $("#history-warning");

  let evaluations;
  try { evaluations = await api("/api/evaluations"); }
  catch (error) { container.replaceChildren(el("p", { className: "empty", text: error.message })); return; }

  if (!evaluations.length) {
    container.replaceChildren(el("p", { className: "empty" }, document.createTextNode("No evaluation history found. "),
      el("a", { text: "Run your first evaluation", attrs: { href: "/evaluate" } }), document.createTextNode(".")));
    return;
  }

  const rows = evaluations.map((e) => {
    const box = el("input", { attrs: { type: "checkbox", value: e.id, "aria-label": `Select ${e.model_filename} on ${e.dataset_filename}` } });
    const row = el("tr", {},
      el("td", { className: "check-cell" }, box),
      el("td", {}, el("a", { text: e.model_filename, attrs: { href: `/evaluations/${e.id}` } })),
      el("td", { text: e.dataset_filename }),
      el("td", {}, el("span", { className: `task task-${e.task_type}`, text: e.task_type[0].toUpperCase() + e.task_type.slice(1) })),
      el("td", { className: "num", text: headlineMetric(e) }),
      el("td", { text: fmtDate(e.evaluated_at) }));
    return { e, box, row };
  });
  const table = el("table", {},
    el("thead", {}, el("tr", {}, el("th", { className: "check-cell" }, el("span", { className: "sr-only", text: "Select" })),
      el("th", { text: "Model" }), el("th", { text: "Dataset" }), el("th", { text: "Task" }), el("th", { className: "num", text: "Headline" }), el("th", { text: "Run on" }))),
    el("tbody", {}, ...rows.map((r) => r.row)));
  container.replaceChildren(el("div", { className: "table-wrap" }, table));

  const selected = () => rows.filter((r) => r.box.checked).map((r) => r.e);
  const update = () => {
    const chosen = selected();
    status.textContent = chosen.length ? `${chosen.length} selected` : "Nothing selected";
    reportButton.disabled = chosen.length < 1;
    const types = new Set(chosen.map((e) => e.task_type));
    const mixed = types.size > 1;
    compareButton.disabled = chosen.length < 2 || mixed;
    warning.hidden = !mixed;
    if (mixed) warning.textContent = "You selected both classification and regression evaluations. Only evaluations of the same task type can be compared. You can still put them in one report.";
  };
  rows.forEach((r) => r.box.addEventListener("change", update));
  compareButton.addEventListener("click", () => { window.location.href = `/compare?ids=${selected().map((e) => e.id).join(",")}`; });
  reportButton.addEventListener("click", () => downloadReport(selected().map((e) => e.id)));
  update();
}

// ---------------------------------------------------------------- admin accounts page

async function initAdminUsers() {
  const list = $("#user-list");
  const form = $("#create-user-form");
  const result = $("#create-user-result");
  let me;
  try { me = await api("/api/me"); } catch (_) { return; }

  const load = async () => {
    try {
      const users = await api("/api/admin/users");
      const activeAdmins = users.filter((u) => u.role === "admin" && u.status === "active").length;
      const table = el("table", {},
        el("thead", {}, el("tr", {}, ...["Name", "Email", "Role", "Status", "Last login", "Actions"].map((h) => el("th", { text: h })))),
        el("tbody", {}, ...users.map((u) => userRow(u, me, activeAdmins, load))));
      list.replaceChildren(el("div", { className: "table-wrap" }, table));
    } catch (error) { list.replaceChildren(el("p", { className: "empty", text: error.message })); }
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(form));
    const button = form.querySelector("button[type=submit]");
    await busy(button, "Creating…", async () => {
      try {
        const user = await postJson("/api/admin/users", data);
        showResult(result, `Created ${user.role} account for ${user.email}.`, "success");
        form.reset(); await load();
      } catch (error) { showResult(result, error.message, "error"); }
    });
  });
  load();
}

function userRow(u, me, activeAdmins, reload) {
  const isMe = u.id === me.id;
  const lastAdmin = u.role === "admin" && u.status === "active" && activeAdmins <= 1;
  const act = async (button, url, body, done) => {
    await busy(button, "Saving…", async () => {
      try { await postJson(url, body); toast(done); await reload(); } catch (error) { toast(error.message); }
    });
  };

  const statusButton = el("button", { className: "small secondary", text: u.status === "active" ? "Suspend" : "Reactivate",
    on: { click: async () => {
      if (u.status === "active" && !await confirmDialog("Suspend this account?", `${u.email} will be logged out and unable to log in until you reactivate it.`, "Suspend")) return;
      act(statusButton, `/api/admin/users/${u.id}/status`, { status: u.status === "active" ? "suspended" : "active" },
        u.status === "active" ? "Account suspended." : "Account reactivated.");
    } } });
  const newRole = u.role === "admin" ? "analyst" : "admin";
  const roleButton = el("button", { className: "small secondary", text: newRole === "admin" ? "Make Admin" : "Make Analyst",
    on: { click: async () => {
      if (!await confirmDialog(`Change role to ${newRole === "admin" ? "Admin" : "Analyst"}?`,
        newRole === "admin" ? `${u.email} will be able to manage all accounts.` : `${u.email} will lose access to the Admin pages.`, "Change role")) return;
      act(roleButton, `/api/admin/users/${u.id}/role`, { role: newRole }, "Role updated.");
    } } });

  let hint = null;
  if (isMe) { statusButton.disabled = roleButton.disabled = true; hint = "You can't change your own account."; }
  else if (lastAdmin) { statusButton.disabled = roleButton.disabled = true; hint = "Last active Admin."; }
  [statusButton, roleButton].forEach((b) => { if (hint) b.title = hint; });

  return el("tr", { className: "user-row" },
    el("td", {}, document.createTextNode(u.name), isMe ? el("span", { className: "you", text: "(you)" }) : null),
    el("td", { text: u.email }),
    el("td", { text: u.role === "admin" ? "Admin" : "Analyst" }),
    el("td", {}, pill(u.status)),
    el("td", { text: u.last_login_at ? fmtDate(u.last_login_at) : "Never" }),
    el("td", {}, el("div", { className: "record-actions" }, statusButton, roleButton)));
}

// ---------------------------------------------------------------- init

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-local-time]").forEach((node) => {
    const iso = node.getAttribute("datetime");
    if (iso) node.textContent = fmtDate(iso);
  });
  document.querySelectorAll(".js-report").forEach((button) =>
    button.addEventListener("click", () => downloadReport(button.dataset.ids.split(","))));

  const page = document.body.dataset.page;
  const pages = { models: initModels, datasets: initDatasets, evaluate: initEvaluate, history: initHistory, "admin-users": initAdminUsers };
  if (pages[page]) pages[page]();
});
