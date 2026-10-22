async function postForm(url, formData) {
  const res = await fetch(url, { method: "POST", body: formData });
  const data = await res.json();
  return { ok: res.ok, data };
}

function showResult(el, ok, data) {
  el.textContent = JSON.stringify(data, null, 2);
  el.className = "result " + (ok ? "success" : "error");
}

document.getElementById("model-upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const formData = new FormData(form);
  const resultEl = document.getElementById("model-upload-result");
  const { ok, data } = await postForm("/api/models/upload", formData);
  showResult(resultEl, ok, data);
  if (ok) form.reset();
});
