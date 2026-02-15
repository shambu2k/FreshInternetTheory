const STORAGE_KEY_BACKEND_BASE_URL = "backendBaseUrl";
const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

function normalizeBaseUrl(raw) {
  const trimmed = String(raw || "").trim();
  if (!trimmed) {
    return DEFAULT_BASE_URL;
  }

  try {
    const parsed = new URL(trimmed);
    return parsed.href.replace(/\/+$/, "");
  } catch (_err) {
    return DEFAULT_BASE_URL;
  }
}

function getBaseUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(
      { [STORAGE_KEY_BACKEND_BASE_URL]: DEFAULT_BASE_URL },
      (items) => {
        if (chrome.runtime.lastError) {
          resolve(DEFAULT_BASE_URL);
          return;
        }
        resolve(normalizeBaseUrl(items[STORAGE_KEY_BACKEND_BASE_URL]));
      }
    );
  });
}

async function proxyBackendRequest({ method = "GET", path = "/", body }) {
  const baseUrl = await getBaseUrl();
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${baseUrl}${normalizedPath}`;

  const init = {
    method,
    headers: {
      Accept: "application/json"
    }
  };

  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  try {
    const response = await fetch(url, init);
    const text = await response.text();
    let data = null;

    if (text) {
      try {
        data = JSON.parse(text);
      } catch (_parseErr) {
        data = { raw: text };
      }
    }

    return {
      ok: response.ok,
      status: response.status,
      data
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      error: err instanceof Error ? err.message : String(err)
    };
  }
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "backendRequest") {
    return false;
  }

  proxyBackendRequest(message.request || {})
    .then(sendResponse)
    .catch((err) => {
      sendResponse({
        ok: false,
        status: 0,
        error: err instanceof Error ? err.message : String(err)
      });
    });

  return true;
});
