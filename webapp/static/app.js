const state = { view: "training", mode: "new", tools: {}, models: [], jobs: [], selectedTool: null, selectedJob: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "toast", 3200);
}

function switchView(view) {
  state.view = view;
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  const titles = { training: "模型訓練", tools: "工具包", jobs: "工作紀錄" };
  $("#page-title").textContent = titles[view];
}

function setMode(mode) {
  state.mode = mode;
  $$(".segmented button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  $$(".new-only").forEach((element) => element.classList.toggle("hidden", mode !== "new"));
  $$(".continue-only").forEach((element) => element.classList.toggle("hidden", mode !== "continue"));
  $("#learning-rate").value = mode === "new" ? "0.0003" : "0.0001";
  $("#timesteps").value = mode === "new" ? "50000" : "25000";
}

async function loadHealth() {
  try {
    await api("/api/health");
    $("#health-dot").classList.add("online");
    $("#health-text").textContent = "系統已連線";
  } catch {
    $("#health-dot").classList.remove("online");
    $("#health-text").textContent = "系統離線";
  }
}

async function loadModels() {
  state.models = await api("/api/models");
  $("#model-count").textContent = state.models.length;
  const select = $("#existing-model");
  select.innerHTML = state.models.length
    ? state.models.map((model) => `<option value="${escapeHtml(model.path)}">${escapeHtml(model.path)} · ${formatBytes(model.size)}</option>`).join("")
    : '<option value="">沒有找到模型</option>';
}

async function loadTools() {
  state.tools = await api("/api/tools");
  const container = $("#package-list");
  container.innerHTML = Object.entries(state.tools).map(([name, tools]) => `
    <div class="package-group"><h4>${escapeHtml(name)} · ${tools.length}</h4>
      ${tools.map((tool) => `<button class="tool-button" data-module="${escapeHtml(tool.module)}">${escapeHtml(tool.module.split(".").slice(1).join("."))}</button>`).join("")}
    </div>`).join("");
  $$(".tool-button").forEach((button) => button.addEventListener("click", () => selectTool(button.dataset.module)));
}

function selectTool(module) {
  state.selectedTool = Object.values(state.tools).flat().find((tool) => tool.module === module);
  if (!state.selectedTool) return;
  $$(".tool-button").forEach((button) => button.classList.toggle("active", button.dataset.module === module));
  $("#tool-empty").classList.add("hidden");
  $("#tool-form").classList.remove("hidden");
  $("#tool-package").textContent = state.selectedTool.package;
  $("#tool-name").textContent = state.selectedTool.module.split(".").slice(1).join(".");
  $("#tool-description").textContent = state.selectedTool.description;
  $("#tool-module").textContent = `python -m ${state.selectedTool.module}`;
  $("#tool-arguments").innerHTML = state.selectedTool.arguments.length
    ? state.selectedTool.arguments.map(argumentField).join("")
    : '<p class="muted">此工具沒有命令列參數。</p>';
}

function argumentField(argument) {
  const name = escapeHtml(argument.display_name);
  const note = [argument.default ? `預設：${argument.default}` : "", argument.help || ""].filter(Boolean).join(" · ");
  if (["store_true", "store_false"].includes(argument.action)) {
    return `<label class="switch field"><input type="checkbox" data-argument="${name}"><span></span>${name}${note ? ` · ${escapeHtml(note)}` : ""}</label>`;
  }
  const numeric = /(steps|seed|count|episodes|fps|size|rate|freq)/i.test(argument.display_name);
  return `<label class="field"><span>${name}${argument.required ? " *" : ""}</span><input data-argument="${name}" type="${numeric ? "number" : "text"}" ${numeric ? 'step="any"' : ""} placeholder="${escapeHtml(note)}" ${argument.required ? "required" : ""}></label>`;
}

async function submitTraining(event) {
  event.preventDefault();
  const modelPath = state.mode === "new" ? $("#new-model-path").value.trim() : $("#existing-model").value;
  if (!modelPath) return toast("請選擇模型。", true);
  const postEnabled = $("#post-enabled").checked;
  const payload = {
    mode: state.mode,
    model_path: modelPath,
    timesteps: Number($("#timesteps").value),
    learning_rate: Number($("#learning-rate").value),
    post_test: {
      enabled: postEnabled,
      test_name: $("#test-name").value.trim() || null,
      episodes: Number($("#episodes").value),
      max_steps: Number($("#max-steps").value),
      record_steps: Number($("#record-steps").value),
      seed: Number($("#record-seed").value),
      fps: Number($("#fps").value),
      screen_size: Number($("#screen-size").value)
    }
  };
  try {
    const job = await api("/api/training", { method: "POST", body: JSON.stringify(payload) });
    state.selectedJob = job.id;
    toast("訓練工作已建立。");
    switchView("jobs");
    await loadJobs();
  } catch (error) { toast(error.message, true); }
}

async function submitTool(event) {
  event.preventDefault();
  const argumentsObject = {};
  $$("#tool-arguments [data-argument]").forEach((input) => {
    argumentsObject[input.dataset.argument] = input.type === "checkbox" ? input.checked : input.value.trim();
  });
  try {
    const job = await api("/api/tools/run", { method: "POST", body: JSON.stringify({ module: state.selectedTool.module, arguments: argumentsObject }) });
    state.selectedJob = job.id;
    toast("工具工作已建立。");
    switchView("jobs");
    await loadJobs();
  } catch (error) { toast(error.message, true); }
}

async function loadJobs() {
  state.jobs = await api("/api/jobs");
  if (!state.selectedJob && state.jobs.length) state.selectedJob = state.jobs[0].id;
  $("#job-list").innerHTML = state.jobs.length ? state.jobs.map((job) => `
    <button class="job-item ${job.id === state.selectedJob ? "active" : ""}" data-job="${job.id}">
      <strong>${escapeHtml(job.name)}</strong><span>${escapeHtml(job.stage)} · ${escapeHtml(job.created_at)}</span>
    </button>`).join("") : '<div class="empty-state"><p>目前沒有工作。</p></div>';
  $$(".job-item").forEach((button) => button.addEventListener("click", () => { state.selectedJob = button.dataset.job; renderSelectedJob(); }));
  renderSelectedJob();
}

function renderSelectedJob() {
  const job = state.jobs.find((item) => item.id === state.selectedJob);
  if (!job) return;
  $$(".job-item").forEach((button) => button.classList.toggle("active", button.dataset.job === job.id));
  $("#job-title").textContent = `${job.name} · ${job.stage}`;
  const badge = $("#job-status");
  badge.textContent = job.status.toUpperCase();
  badge.className = `status-badge ${job.status}`;
  $("#job-logs").textContent = job.logs.length ? job.logs.join("\n") : "工作已排入佇列…";
  $("#job-logs").scrollTop = $("#job-logs").scrollHeight;
  $("#cancel-job").classList.toggle("hidden", !["queued", "running"].includes(job.status));
}

async function cancelSelectedJob() {
  if (!state.selectedJob) return;
  try { await api(`/api/jobs/${state.selectedJob}/cancel`, { method: "POST" }); await loadJobs(); }
  catch (error) { toast(error.message, true); }
}

function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character]); }
function formatBytes(value) { const units = ["B", "KB", "MB", "GB"]; let size = value; let index = 0; while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; } return `${size.toFixed(index ? 1 : 0)} ${units[index]}`; }

async function refreshAll() {
  try { await Promise.all([loadHealth(), loadModels(), loadTools(), loadJobs()]); }
  catch (error) { toast(error.message, true); }
}

document.addEventListener("DOMContentLoaded", () => {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$(".segmented button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $("#post-enabled").addEventListener("change", (event) => $$("#post-fields input").forEach((input) => input.disabled = !event.target.checked));
  $("#training-form").addEventListener("submit", submitTraining);
  $("#tool-form").addEventListener("submit", submitTool);
  $("#cancel-job").addEventListener("click", cancelSelectedJob);
  $("#refresh-button").addEventListener("click", refreshAll);
  refreshAll();
  setInterval(() => { loadJobs().catch(() => {}); loadHealth().catch(() => {}); }, 1500);
});
