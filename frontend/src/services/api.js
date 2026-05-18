// Single place for all backend communication (per Web Module rules).

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";

async function asJson(response) {
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const message =
      (data && (data.error || (data.errors && data.errors.join(", ")))) ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

// Health check.
export async function checkBackend() {
  const response = await fetch(`${API_BASE}/api/health`);
  return response.json();
}

// Upload a CSV file for authoritative server-side validation/parsing.
export async function uploadCsv(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: form,
  });
  return asJson(response);
}

// Run (mock) inference on edited rows.
export async function runPrediction(rows) {
  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows }),
  });
  return asJson(response);
}

// Trigger (mock) retraining.
export async function runTraining(payload = {}) {
  const response = await fetch(`${API_BASE}/api/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return asJson(response);
}

// Persist user-edited predictions + write audit trail.
export async function submitFeedback(records, before = [], user = "anonymous") {
  const response = await fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records, before, user }),
  });
  return asJson(response);
}

export async function getHistory() {
  const response = await fetch(`${API_BASE}/api/history`);
  return asJson(response);
}

export async function getHistoryItem(id) {
  const response = await fetch(`${API_BASE}/api/history/${id}`);
  return asJson(response);
}
