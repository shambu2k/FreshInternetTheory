const STORAGE_KEY_BACKEND_BASE_URL = "backendBaseUrl";
const STORAGE_KEY_AUTOSKIP_AI = "autoSkipAiReels";
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
  status.style.color = isError ? "#ff9a93" : "#84daba";
}

function loadOptions() {
  chrome.storage.sync.get(
    {
      [STORAGE_KEY_BACKEND_BASE_URL]: DEFAULT_BASE_URL,
      [STORAGE_KEY_AUTOSKIP_AI]: false
    },
    (items) => {
      const urlInput = document.getElementById("backend-url");
      const autoSkipInput = document.getElementById("auto-skip-ai");

      urlInput.value = items[STORAGE_KEY_BACKEND_BASE_URL] || DEFAULT_BASE_URL;
      autoSkipInput.checked = Boolean(items[STORAGE_KEY_AUTOSKIP_AI]);
    }
  );
}

function saveOptions() {
  const urlInput = document.getElementById("backend-url");
  const autoSkipInput = document.getElementById("auto-skip-ai");

  const normalized = normalizeBaseUrl(urlInput.value);
  if (!normalized) {
    setStatus("Enter a valid URL including protocol (http/https).", true);
    return;
  }

  chrome.storage.sync.set(
    {
      [STORAGE_KEY_BACKEND_BASE_URL]: normalized,
      [STORAGE_KEY_AUTOSKIP_AI]: Boolean(autoSkipInput.checked)
    },
    () => {
      if (chrome.runtime.lastError) {
        setStatus(chrome.runtime.lastError.message || "Failed to save settings.", true);
        return;
      }
      urlInput.value = normalized;
      setStatus("Settings saved.", false);
    }
  );
}

document.addEventListener("DOMContentLoaded", () => {
  loadOptions();
  document.getElementById("save-btn").addEventListener("click", saveOptions);
});
