const STORAGE_KEY_BACKEND_BASE_URL = "backendBaseUrl";
const DEFAULT_BASE_URL = "http://localhost:8000";

function normalizeBaseUrl(raw) {
  const trimmed = String(raw || "").trim();
  if (!trimmed) {
    return null;
  }

  try {
    const parsed = new URL(trimmed);
    return parsed.href.replace(/\/+$/, "");
  } catch (_err) {
    return null;
  }
}

function setStatus(message, isError) {
  const status = document.getElementById("status");
  status.textContent = message;
  status.style.color = isError ? "#b00020" : "#137333";
}

function loadOptions() {
  chrome.storage.sync.get(
    { [STORAGE_KEY_BACKEND_BASE_URL]: DEFAULT_BASE_URL },
    (items) => {
      const input = document.getElementById("backend-url");
      input.value = items[STORAGE_KEY_BACKEND_BASE_URL] || DEFAULT_BASE_URL;
    }
  );
}

function saveOptions() {
  const input = document.getElementById("backend-url");
  const normalized = normalizeBaseUrl(input.value);
  if (!normalized) {
    setStatus("Enter a valid URL including protocol (http/https).", true);
    return;
  }

  chrome.storage.sync.set({ [STORAGE_KEY_BACKEND_BASE_URL]: normalized }, () => {
    if (chrome.runtime.lastError) {
      setStatus(chrome.runtime.lastError.message || "Failed to save URL.", true);
      return;
    }
    input.value = normalized;
    setStatus("Saved.", false);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadOptions();
  document.getElementById("save-btn").addEventListener("click", saveOptions);
});
