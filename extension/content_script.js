(() => {
  const CACHE_TTL_MS = {
    known: 30 * 1000,
    unknown: 2 * 60 * 1000
  };
  const VOTE_COOLDOWN_MS = 10 * 1000;
  const AUTO_SKIP_ATTEMPT_GAP_MS = 10 * 1000;
  const AUTO_SKIP_RETRY_COUNT = 8;
  const AUTO_SKIP_RETRY_INTERVAL_MS = 450;
  const STORAGE_KEY_AUTOSKIP_AI = "autoSkipAiReels";
  const OVERLAY_HOST_ID = "fit-reel-overlay-host";

  const reelCache = new Map();
  const voteCooldowns = new Map();
  const autoSkipAttempts = new Map();
  const autoSkipSessions = new Map();

  let overlayHost = null;
  let overlayContent = null;
  let currentReelId = null;
  let lastHref = location.href;
  let currentLoadToken = 0;
  let observerStarted = false;
  let historyPatched = false;
  let detectTimer = null;
  let autoSkipAiReels = false;

  function stableHash(input) {
    let hash = 2166136261;
    for (let i = 0; i < input.length; i += 1) {
      hash ^= input.charCodeAt(i);
      hash +=
        (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return (hash >>> 0).toString(16);
  }

  function parseReelIdFromPath(pathname) {
    const patterns = [/\/reel\/([A-Za-z0-9_-]+)/i, /\/reels\/([A-Za-z0-9_-]+)/i];
    for (const pattern of patterns) {
      const match = pathname.match(pattern);
      if (match && match[1]) {
        return match[1];
      }
    }
    return null;
  }

  function parseReelIdFromUrl(rawUrl) {
    try {
      const parsed = new URL(rawUrl, location.origin);
      return parseReelIdFromPath(parsed.pathname);
    } catch (_err) {
      return null;
    }
  }

  function isReelContext() {
    if (/\/reel(s)?\//i.test(location.pathname)) {
      return true;
    }

    return Boolean(
      document.querySelector(
        'link[rel="canonical"][href*="/reel/"], link[rel="canonical"][href*="/reels/"], meta[property="og:url"][content*="/reel/"], meta[property="og:url"][content*="/reels/"], a[href*="/reel/"], a[href*="/reels/"]'
      )
    );
  }

  function extractReelIdFromDom() {
    const candidates = [];

    const canonical = document.querySelector('link[rel="canonical"]');
    if (canonical && canonical.href) {
      candidates.push(canonical.href);
    }

    const ogUrl = document.querySelector('meta[property="og:url"]');
    if (ogUrl && ogUrl.content) {
      candidates.push(ogUrl.content);
    }

    const anchors = document.querySelectorAll(
      'a[href*="/reel/"], a[href*="/reels/"]'
    );
    for (let i = 0; i < anchors.length && i < 25; i += 1) {
      const href = anchors[i].getAttribute("href");
      if (!href) {
        continue;
      }
      candidates.push(href);
    }

    for (const candidate of candidates) {
      const reelId = parseReelIdFromUrl(candidate);
      if (reelId) {
        return reelId;
      }
    }

    return null;
  }

  function getReelId() {
    const fromPath = parseReelIdFromPath(location.pathname);
    if (fromPath) {
      return fromPath;
    }

    const fromDom = extractReelIdFromDom();
    if (fromDom) {
      return fromDom;
    }

    if (!isReelContext()) {
      return null;
    }

    return `href_${stableHash(location.href)}`;
  }

  function ensureOverlay() {
    if (!document.documentElement) {
      return;
    }

    if (overlayHost && overlayHost.isConnected && overlayContent) {
      return;
    }

    overlayHost = document.getElementById(OVERLAY_HOST_ID);
    if (!overlayHost) {
      overlayHost = document.createElement("div");
      overlayHost.id = OVERLAY_HOST_ID;
      overlayHost.style.position = "fixed";
      overlayHost.style.top = "16px";
      overlayHost.style.right = "64px";
      overlayHost.style.zIndex = "2147483647";
      overlayHost.style.maxWidth = "340px";
      overlayHost.style.pointerEvents = "none";
      document.documentElement.appendChild(overlayHost);
    }

    if (!overlayHost.shadowRoot) {
      const shadow = overlayHost.attachShadow({ mode: "open" });

      const style = document.createElement("style");
      style.textContent = `
        .fit-card {
          pointer-events: auto;
          font-family: 'Courier New', Courier, monospace, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          width: 332px;
          background:
            radial-gradient(120% 100% at 100% 0%, rgba(190, 151, 81, 0.16), transparent 58%),
            radial-gradient(140% 120% at 0% 100%, rgba(53, 72, 88, 0.2), transparent 62%),
            linear-gradient(165deg, rgba(12, 14, 18, 0.96), rgba(20, 22, 29, 0.96));
          color: #f0f1f5;
          border-radius: 14px;
          border: 1px solid rgba(176, 144, 89, 0.38);
          box-shadow: 0 18px 42px rgba(5, 5, 7, 0.52);
          padding: 14px;
          backdrop-filter: blur(6px);
        }
        .fit-title {
          font-size: 17px;
          font-weight: 760;
          letter-spacing: 0.6px;
          text-transform: uppercase;
          margin-bottom: 2px;
        }
        .fit-subtitle {
          font-size: 12px;
          color: #b8bfcb;
          margin-bottom: 4px;
          line-height: 1.4;
        }
        .fit-mode {
          font-size: 11px;
          letter-spacing: 0.45px;
          text-transform: uppercase;
          color: #b99767;
          margin-bottom: 10px;
        }
        .fit-meta {
          font-size: 11px;
          opacity: 0.78;
          margin-bottom: 10px;
          color: #9ca3b2;
          word-break: break-all;
        }
        .fit-stat {
          font-size: 13px;
          margin-bottom: 6px;
          line-height: 1.3;
          padding: 6px 8px;
          border: 1px solid rgba(154, 163, 179, 0.22);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.03);
        }
        .fit-question {
          margin-top: 10px;
          font-size: 13px;
          line-height: 1.35;
          color: #c6ccd8;
        }
        .fit-buttons {
          margin-top: 12px;
          display: flex;
          gap: 8px;
        }
        .fit-btn {
          flex: 1;
          border: 1px solid transparent;
          border-radius: 8px;
          padding: 9px 10px;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 0.2px;
          cursor: pointer;
          transition: transform 120ms ease, opacity 120ms ease, border-color 120ms ease;
        }
        .fit-btn:hover:not(:disabled) {
          opacity: 0.92;
          transform: translateY(-1px);
        }
        .fit-btn:disabled {
          cursor: not-allowed;
          opacity: 0.5;
        }
        .fit-btn-ai {
          background: linear-gradient(165deg, #983d42, #7c3036);
          border-color: rgba(224, 148, 148, 0.4);
          color: #fff4f5;
        }
        .fit-btn-not-ai {
          background: linear-gradient(165deg, #3b4f67, #2e3f55);
          border-color: rgba(150, 172, 198, 0.42);
          color: #f2f7ff;
        }
        .fit-status {
          margin-top: 8px;
          font-size: 12px;
          color: #c3cad6;
          line-height: 1.35;
        }
        .fit-verdict {
          margin-bottom: 10px;
        }
        .fit-badge {
          display: inline-block;
          border-radius: 999px;
          padding: 5px 10px;
          font-size: 12px;
          font-weight: 800;
          letter-spacing: 0.35px;
          text-transform: uppercase;
          border: 1px solid transparent;
        }
        .fit-badge-ai {
          background: rgba(172, 62, 67, 0.22);
          border-color: rgba(221, 132, 132, 0.52);
          color: #ffe0e2;
        }
        .fit-badge-not-ai {
          background: rgba(73, 112, 88, 0.25);
          border-color: rgba(138, 192, 156, 0.45);
          color: #dbf5e8;
        }
        .fit-badge-pending {
          background: rgba(164, 128, 71, 0.26);
          border-color: rgba(216, 184, 130, 0.46);
          color: #f4e6ce;
        }
        .fit-badge-failed {
          background: rgba(133, 84, 56, 0.3);
          border-color: rgba(196, 132, 93, 0.5);
          color: #ffdfc9;
        }
        .fit-badge-neutral {
          background: rgba(103, 111, 126, 0.24);
          border-color: rgba(162, 173, 193, 0.5);
          color: #e9edf5;
        }
        .fit-accordion {
          margin-top: 8px;
          border: 1px solid rgba(193, 170, 132, 0.35);
          border-radius: 8px;
          padding: 2px 8px 7px;
          background: rgba(255, 255, 255, 0.03);
        }
        .fit-accordion summary {
          cursor: pointer;
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 0.2px;
          padding: 5px 0;
          list-style: none;
          outline: none;
          color: #ddc9a5;
        }
        .fit-accordion summary::-webkit-details-marker {
          display: none;
        }
        .fit-accordion summary::before {
          content: "▸";
          margin-right: 6px;
          opacity: 0.9;
        }
        .fit-accordion[open] summary::before {
          content: "▾";
        }
        .fit-reason-line {
          margin-top: 4px;
          font-size: 12px;
          line-height: 1.35;
          color: #d8dde8;
        }
      `;

      overlayContent = document.createElement("div");
      overlayContent.className = "fit-card";
      shadow.append(style, overlayContent);
    } else {
      overlayContent = overlayHost.shadowRoot.querySelector(".fit-card");
    }
  }

  function showOverlay() {
    ensureOverlay();
    if (overlayHost) {
      overlayHost.style.display = "block";
    }
  }

  function hideOverlay() {
    if (overlayHost) {
      overlayHost.style.display = "none";
    }
  }

  function createNode(tagName, className, textContent) {
    const node = document.createElement(tagName);
    if (className) {
      node.className = className;
    }
    if (textContent !== undefined) {
      node.textContent = textContent;
    }
    return node;
  }

  function buildBaseCard(reelId) {
    ensureOverlay();
    if (!overlayContent) {
      return null;
    }

    overlayContent.textContent = "";

    const title = createNode("div", "fit-title", "Nightwire");
    const subtitle = createNode("div", "fit-subtitle", "Fresh Internet Theory");
    const mode = createNode(
      "div",
      "fit-mode",
      autoSkipAiReels ? "Auto-skip AI: On" : "Auto-skip AI: Off"
    );
    const meta = createNode("div", "fit-meta", `reel_id: ${reelId}`);

    overlayContent.append(title, subtitle, mode, meta);
    return overlayContent;
  }

  function getCooldownRemainingMs(reelId) {
    const expiresAt = voteCooldowns.get(reelId) || 0;
    return Math.max(0, expiresAt - Date.now());
  }

  function startCooldown(reelId) {
    voteCooldowns.set(reelId, Date.now() + VOTE_COOLDOWN_MS);
  }

  function scheduleCooldownRefresh(reelId) {
    const waitMs = getCooldownRemainingMs(reelId);
    if (waitMs <= 0) {
      return;
    }
    window.setTimeout(() => {
      if (currentReelId !== reelId) {
        return;
      }
      loadAndRenderReel(reelId, { showLoading: false, forceNetwork: false });
    }, waitMs + 60);
  }

  function setCacheEntry(reelId, entry) {
    reelCache.set(reelId, {
      lastFetchAt: Date.now(),
      lastKnownStatus: entry.lastKnownStatus,
      counts: entry.counts || { aiCount: 0, notAiCount: 0 },
      verdict: entry.verdict || null
    });
  }

  function getFreshCacheEntry(reelId) {
    const cached = reelCache.get(reelId);
    if (!cached) {
      return null;
    }

    const ttl = cached.lastKnownStatus === "unknown"
      ? CACHE_TTL_MS.unknown
      : CACHE_TTL_MS.known;
    if (Date.now() - cached.lastFetchAt > ttl) {
      return null;
    }
    return cached;
  }

  function computeCounts(reelData) {
    const responses = Array.isArray(reelData?.user_responses)
      ? reelData.user_responses
      : [];

    let aiCount = 0;
    let notAiCount = 0;
    for (const response of responses) {
      if (response && response.label === "ai") {
        aiCount += 1;
      } else if (response && response.label === "not_ai") {
        notAiCount += 1;
      }
    }

    return { aiCount, notAiCount };
  }

  function truncateText(value, maxLen = 180) {
    const text = String(value || "").trim();
    if (!text) {
      return "";
    }
    if (text.length <= maxLen) {
      return text;
    }
    return `${text.slice(0, maxLen - 1)}…`;
  }

  function parseBoolean(value) {
    if (typeof value === "boolean") {
      return value;
    }
    if (typeof value === "string") {
      const normalized = value.trim().toLowerCase();
      if (["true", "yes", "1"].includes(normalized)) {
        return true;
      }
      if (["false", "no", "0"].includes(normalized)) {
        return false;
      }
    }
    return null;
  }

  function parseConfidence(value) {
    if (value === null || value === undefined || value === "") {
      return null;
    }
    const num = Number(value);
    if (!Number.isFinite(num)) {
      return null;
    }
    const normalized = num <= 1 ? num * 100 : num;
    if (normalized < 0) {
      return 0;
    }
    if (normalized > 100) {
      return 100;
    }
    return Math.round(normalized);
  }

  function safeJsonParse(input) {
    if (typeof input !== "string") {
      return null;
    }
    const trimmed = input.trim();
    if (!trimmed) {
      return null;
    }
    try {
      return JSON.parse(trimmed);
    } catch (_err) {
      return null;
    }
  }

  function extractAnalysisCandidate(node, depth = 0) {
    if (depth > 4 || node === null || node === undefined) {
      return null;
    }

    if (typeof node === "string") {
      const parsed = safeJsonParse(node);
      if (parsed) {
        return extractAnalysisCandidate(parsed, depth + 1);
      }
      return null;
    }

    if (Array.isArray(node)) {
      for (const item of node.slice(0, 5)) {
        const found = extractAnalysisCandidate(item, depth + 1);
        if (found) {
          return found;
        }
      }
      return null;
    }

    if (typeof node !== "object") {
      return null;
    }

    if (
      Object.prototype.hasOwnProperty.call(node, "is_likely_ai_generated") ||
      Object.prototype.hasOwnProperty.call(node, "confidence") ||
      Object.prototype.hasOwnProperty.call(node, "key_evidence")
    ) {
      return node;
    }

    const preferredKeys = [
      "raw_analysis",
      "analysis",
      "result",
      "output",
      "data",
      "response"
    ];

    for (const key of preferredKeys) {
      if (!Object.prototype.hasOwnProperty.call(node, key)) {
        continue;
      }
      const found = extractAnalysisCandidate(node[key], depth + 1);
      if (found) {
        return found;
      }
    }

    return null;
  }

  function extractReasonLines(candidate, fallbackReason) {
    const lines = [];

    const summaryFields = ["summary", "reason", "explanation", "message"];
    for (const field of summaryFields) {
      if (typeof candidate?.[field] === "string" && candidate[field].trim()) {
        lines.push(truncateText(candidate[field]));
        break;
      }
    }

    const evidence = Array.isArray(candidate?.key_evidence)
      ? candidate.key_evidence
      : Array.isArray(candidate?.evidence)
        ? candidate.evidence
        : [];

    for (const item of evidence.slice(0, 4)) {
      if (typeof item === "string") {
        lines.push(truncateText(item));
        continue;
      }
      if (!item || typeof item !== "object") {
        continue;
      }

      const category = typeof item.category === "string" && item.category.trim()
        ? `[${item.category.trim()}] `
        : "";
      const descriptionRaw =
        item.description || item.reason || item.evidence || item.detail || "";
      const description = typeof descriptionRaw === "string"
        ? descriptionRaw.trim()
        : "";
      const timestamp = typeof item.timestamp === "string" && item.timestamp.trim()
        ? ` @ ${item.timestamp.trim()}`
        : "";

      const composed = `${category}${description}${timestamp}`.trim();
      if (composed) {
        lines.push(truncateText(composed));
      }
    }

    if (lines.length === 0 && fallbackReason) {
      lines.push(truncateText(fallbackReason));
    }

    return lines.slice(0, 5);
  }

  function extractBackendVerdict(reelData) {
    const analysis = reelData?.analysis && typeof reelData.analysis === "object"
      ? reelData.analysis
      : {};
    const status = typeof analysis.status === "string" ? analysis.status : "";
    const rawTwelveLabs = analysis.twelvelabs ?? reelData?.["12labs_ai_analysis"];
    const candidate = extractAnalysisCandidate(rawTwelveLabs);

    if (candidate) {
      const isLikelyAi = parseBoolean(
        candidate.is_likely_ai_generated ??
        candidate.is_ai_generated ??
        candidate.ai_generated
      );
      const confidence = parseConfidence(
        candidate.confidence ?? candidate.score ?? candidate.probability
      );
      const confidenceText = confidence === null ? "" : ` (${confidence}%)`;
      const reasonLines = extractReasonLines(candidate, analysis.last_error);

      if (isLikelyAi === true) {
        return {
          tone: "ai",
          badgeText: `Likely AI${confidenceText}`,
          reasonLines
        };
      }

      if (isLikelyAi === false) {
        return {
          tone: "not-ai",
          badgeText: `Likely Not AI${confidenceText}`,
          reasonLines
        };
      }

      return {
        tone: "neutral",
        badgeText: `Analysis available${confidenceText}`,
        reasonLines
      };
    }

    if (status === "queued") {
      return {
        tone: "pending",
        badgeText: "Analysis queued",
        reasonLines: []
      };
    }

    if (status === "not_started") {
      return {
        tone: "pending",
        badgeText: "Analysis not started",
        reasonLines: []
      };
    }

    if (status === "failed") {
      return {
        tone: "failed",
        badgeText: "Analysis failed",
        reasonLines: extractReasonLines(null, analysis.last_error || "No error details available.")
      };
    }

    return {
      tone: "neutral",
      badgeText: "No analysis yet",
      reasonLines: []
    };
  }

  function loadClientSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get({ [STORAGE_KEY_AUTOSKIP_AI]: false }, (items) => {
        if (chrome.runtime.lastError) {
          autoSkipAiReels = false;
          resolve();
          return;
        }
        autoSkipAiReels = Boolean(items[STORAGE_KEY_AUTOSKIP_AI]);
        resolve();
      });
    });
  }

  function watchClientSettings() {
    chrome.storage.onChanged.addListener((changes, areaName) => {
      if (areaName !== "sync") {
        return;
      }
      if (!Object.prototype.hasOwnProperty.call(changes, STORAGE_KEY_AUTOSKIP_AI)) {
        return;
      }
      autoSkipAiReels = Boolean(changes[STORAGE_KEY_AUTOSKIP_AI].newValue);
      if (currentReelId) {
        loadAndRenderReel(currentReelId, { showLoading: false, forceNetwork: false });
      }
    });
  }

  function isVisibleElement(element) {
    if (!(element instanceof HTMLElement)) {
      return false;
    }
    const rect = element.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) {
      return false;
    }
    if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) {
      return false;
    }
    const style = window.getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
      return false;
    }
    const cx = Math.max(0, Math.min(window.innerWidth - 1, Math.round(rect.left + rect.width / 2)));
    const cy = Math.max(0, Math.min(window.innerHeight - 1, Math.round(rect.top + rect.height / 2)));
    const topAtCenter = document.elementFromPoint(cx, cy);
    if (
      topAtCenter &&
      !element.contains(topAtCenter) &&
      !(topAtCenter instanceof HTMLElement && topAtCenter.contains(element))
    ) {
      return false;
    }
    return true;
  }

  function triggerNativeLikeClick(element) {
    if (!(element instanceof HTMLElement)) {
      return false;
    }
    element.focus({ preventScroll: true });
    const rect = element.getBoundingClientRect();
    const cx = Math.max(0, Math.min(window.innerWidth - 1, Math.round(rect.left + rect.width / 2)));
    const cy = Math.max(0, Math.min(window.innerHeight - 1, Math.round(rect.top + rect.height / 2)));
    const centerElement = document.elementFromPoint(cx, cy);

    const targets = [element];
    if (centerElement && centerElement instanceof HTMLElement && centerElement !== element) {
      targets.push(centerElement);
    }

    for (const target of targets) {
      if (typeof PointerEvent === "function") {
        target.dispatchEvent(
          new PointerEvent("pointerdown", {
            view: window,
            bubbles: true,
            cancelable: true,
            pointerType: "mouse",
            clientX: cx,
            clientY: cy
          })
        );
        target.dispatchEvent(
          new PointerEvent("pointerup", {
            view: window,
            bubbles: true,
            cancelable: true,
            pointerType: "mouse",
            clientX: cx,
            clientY: cy
          })
        );
      }

      target.dispatchEvent(
        new MouseEvent("mousedown", {
          view: window,
          bubbles: true,
          cancelable: true,
          clientX: cx,
          clientY: cy
        })
      );
      target.dispatchEvent(
        new MouseEvent("mouseup", {
          view: window,
          bubbles: true,
          cancelable: true,
          clientX: cx,
          clientY: cy
        })
      );
      target.dispatchEvent(
        new MouseEvent("click", {
          view: window,
          bubbles: true,
          cancelable: true,
          clientX: cx,
          clientY: cy
        })
      );
    }

    if (typeof element.click === "function") {
      element.click();
    }
    if (
      centerElement &&
      centerElement instanceof HTMLElement &&
      centerElement !== element &&
      typeof centerElement.click === "function"
    ) {
      centerElement.click();
    }
    return true;
  }

  function attemptAutoSkipNavigation() {
    const selectorsInPriority = [
      '[role="button"][aria-label="Next Card"]',
      '[aria-label="Next Card"]',
      '[role="button"][aria-label*="Next"]',
      'button[aria-label*="Next"]',
      'a[aria-label*="Next"]',
      'div[role="button"]'
    ];

    let nextTarget = null;
    for (const selector of selectorsInPriority) {
      const matches = Array.from(document.querySelectorAll(selector)).filter((element) => {
        if (element.closest(`#${OVERLAY_HOST_ID}`)) {
          return false;
        }
        if (element.hasAttribute("disabled") || element.getAttribute("aria-disabled") === "true") {
          return false;
        }
        if (!isVisibleElement(element)) {
          return false;
        }

        if (selector === 'div[role="button"]') {
          const aria = element.getAttribute("aria-label") || "";
          const title = element.getAttribute("title") || "";
          const text = element.textContent || "";
          const combined = `${aria} ${title} ${text}`.toLowerCase();
          return /(next card|next reel|next video|\bnext\b)/.test(combined);
        }

        return true;
      });

      if (matches.length > 0) {
        nextTarget = matches[0];
        break;
      }
    }

    if (nextTarget) {
      triggerNativeLikeClick(nextTarget);
      return true;
    }

    const dispatchKey = (key, code) => {
      const init = { key, code, bubbles: true, cancelable: true };
      document.dispatchEvent(new KeyboardEvent("keydown", init));
      document.dispatchEvent(new KeyboardEvent("keyup", init));
      if (document.body) {
        document.body.dispatchEvent(new KeyboardEvent("keydown", init));
        document.body.dispatchEvent(new KeyboardEvent("keyup", init));
      }
      const active = document.activeElement;
      if (active && active !== document.body && active !== document.documentElement) {
        active.dispatchEvent(new KeyboardEvent("keydown", init));
        active.dispatchEvent(new KeyboardEvent("keyup", init));
      }
      window.dispatchEvent(new KeyboardEvent("keydown", init));
      window.dispatchEvent(new KeyboardEvent("keyup", init));
    };

    dispatchKey("PageDown", "PageDown");
    dispatchKey("ArrowDown", "ArrowDown");
    window.scrollBy({
      top: Math.max(320, Math.round(window.innerHeight * 0.8)),
      behavior: "smooth"
    });
    return false;
  }

  function clearAutoSkipSession(reelId) {
    const session = autoSkipSessions.get(reelId);
    if (session && session.timer) {
      clearTimeout(session.timer);
    }
    autoSkipSessions.delete(reelId);
  }

  function runAutoSkipAttempt(reelId, session, attemptIndex) {
    if (!session || session.cancelled) {
      clearAutoSkipSession(reelId);
      return;
    }
    if (!autoSkipAiReels || currentReelId !== reelId) {
      clearAutoSkipSession(reelId);
      return;
    }

    const clicked = attemptAutoSkipNavigation();

    if (currentReelId !== reelId) {
      clearAutoSkipSession(reelId);
      return;
    }
    if (attemptIndex >= AUTO_SKIP_RETRY_COUNT - 1) {
      clearAutoSkipSession(reelId);
      renderKnown(
        reelId,
        reelCache.get(reelId)?.counts || { aiCount: 0, notAiCount: 0 },
        reelCache.get(reelId)?.verdict || null,
        "Auto-skip couldn't move this reel. Try manual next once.",
        false
      );
      return;
    }

    const delay = clicked ? AUTO_SKIP_RETRY_INTERVAL_MS : AUTO_SKIP_RETRY_INTERVAL_MS + 120;
    session.timer = window.setTimeout(() => {
      runAutoSkipAttempt(reelId, session, attemptIndex + 1);
    }, delay);
  }

  function maybeAutoSkipKnownReel(reelId, counts, verdict) {
    if (!autoSkipAiReels || !verdict || verdict.tone !== "ai") {
      return;
    }

    const lastAttempt = autoSkipAttempts.get(reelId) || 0;
    if (Date.now() - lastAttempt < AUTO_SKIP_ATTEMPT_GAP_MS) {
      return;
    }

    autoSkipAttempts.set(reelId, Date.now());
    clearAutoSkipSession(reelId);
    renderKnown(
      reelId,
      counts,
      verdict,
      "Auto-skip enabled. Moving to next reel...",
      true
    );
    const session = { cancelled: false, timer: null };
    autoSkipSessions.set(reelId, session);
    session.timer = window.setTimeout(() => {
      runAutoSkipAttempt(reelId, session, 0);
    }, 220);
  }

  function backendRequest(method, path, body) {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage(
        {
          type: "backendRequest",
          request: { method, path, body }
        },
        (response) => {
          if (chrome.runtime.lastError) {
            resolve({
              ok: false,
              status: 0,
              error: chrome.runtime.lastError.message
            });
            return;
          }
          resolve(
            response || {
              ok: false,
              status: 0,
              error: "No response from service worker"
            }
          );
        }
      );
    });
  }

  async function fetchReelState(reelId, options = {}) {
    const useCache = !options.forceNetwork;
    if (useCache) {
      const cached = getFreshCacheEntry(reelId);
      if (cached) {
        return {
          kind: cached.lastKnownStatus,
          counts: cached.counts,
          verdict: cached.verdict,
          fromCache: true
        };
      }
    }

    const result = await backendRequest("GET", `/reels/${encodeURIComponent(reelId)}`);

    if (result.ok && result.status === 200 && result.data && result.data.reel_data) {
      const counts = computeCounts(result.data.reel_data);
      const verdict = extractBackendVerdict(result.data.reel_data);
      setCacheEntry(reelId, { lastKnownStatus: "known", counts, verdict });
      return { kind: "known", counts, verdict, fromCache: false };
    }

    if (
      result.status === 404 ||
      (result.ok && (!result.data || !result.data.reel_data))
    ) {
      setCacheEntry(reelId, { lastKnownStatus: "unknown" });
      return { kind: "unknown", counts: { aiCount: 0, notAiCount: 0 }, fromCache: false };
    }

    return { kind: "offline", error: result.error || `HTTP ${result.status}` };
  }

  function renderLoading(reelId) {
    showOverlay();
    const card = buildBaseCard(reelId);
    if (!card) {
      return;
    }
    card.append(createNode("div", "fit-status", "Loading..."));
  }

  function renderOffline(reelId) {
    showOverlay();
    const card = buildBaseCard(reelId);
    if (!card) {
      return;
    }

    const status = createNode("div", "fit-status", "Backend offline");
    const buttons = createNode("div", "fit-buttons");
    const aiBtn = createNode("button", "fit-btn fit-btn-ai", "Mark AI");
    const notAiBtn = createNode("button", "fit-btn fit-btn-not-ai", "Not AI");
    aiBtn.disabled = true;
    notAiBtn.disabled = true;
    buttons.append(aiBtn, notAiBtn);
    card.append(status, buttons);
  }

  function renderVerdict(card, verdict) {
    if (!card || !verdict) {
      return;
    }

    const verdictWrap = createNode("div", "fit-verdict");
    const toneClass = `fit-badge-${verdict.tone || "neutral"}`;
    verdictWrap.append(createNode("div", `fit-badge ${toneClass}`, verdict.badgeText));

    if (Array.isArray(verdict.reasonLines) && verdict.reasonLines.length > 0) {
      const details = createNode("details", "fit-accordion");
      const summary = createNode("summary", "", "Why this verdict?");
      details.append(summary);

      for (const line of verdict.reasonLines) {
        details.append(createNode("div", "fit-reason-line", line));
      }
      verdictWrap.append(details);
    }

    card.append(verdictWrap);
  }

  function renderKnown(reelId, counts, verdict, statusMessage, forceDisableButtons) {
    showOverlay();
    const card = buildBaseCard(reelId);
    if (!card) {
      return;
    }

    renderVerdict(card, verdict);

    card.append(
      createNode("div", "fit-stat", `AI votes: ${counts.aiCount}`),
      createNode("div", "fit-stat", `Not AI votes: ${counts.notAiCount}`),
      createNode("div", "fit-question", "How do you rate this reel?")
    );

    const buttons = createNode("div", "fit-buttons");
    const aiBtn = createNode("button", "fit-btn fit-btn-ai", "Mark AI");
    const notAiBtn = createNode("button", "fit-btn fit-btn-not-ai", "Not AI");

    const cooldownMs = getCooldownRemainingMs(reelId);
    const disabled = forceDisableButtons || cooldownMs > 0;
    aiBtn.disabled = disabled;
    notAiBtn.disabled = disabled;

    aiBtn.addEventListener("click", () => {
      handleKnownVote(reelId, "ai");
    });
    notAiBtn.addEventListener("click", () => {
      handleKnownVote(reelId, "not_ai");
    });

    buttons.append(aiBtn, notAiBtn);
    card.append(buttons);

    let finalStatus = statusMessage || "";
    if (!finalStatus && cooldownMs > 0) {
      finalStatus = `Vote cooldown: ${Math.ceil(cooldownMs / 1000)}s`;
    }
    if (finalStatus) {
      card.append(createNode("div", "fit-status", finalStatus));
    }
  }

  function renderUnknown(reelId, statusMessage, forceDisableButtons) {
    showOverlay();
    const card = buildBaseCard(reelId);
    if (!card) {
      return;
    }

    card.append(
      createNode("div", "fit-question", "Not rated yet. Is this AI-generated?")
    );

    const buttons = createNode("div", "fit-buttons");
    const aiBtn = createNode("button", "fit-btn fit-btn-ai", "Mark AI");
    const notAiBtn = createNode("button", "fit-btn fit-btn-not-ai", "Not AI");

    const cooldownMs = getCooldownRemainingMs(reelId);
    const disabled = forceDisableButtons || cooldownMs > 0;
    aiBtn.disabled = disabled;
    notAiBtn.disabled = disabled;

    aiBtn.addEventListener("click", () => {
      handleUnknownVote(reelId, "ai");
    });
    notAiBtn.addEventListener("click", () => {
      handleUnknownVote(reelId, "not_ai");
    });

    buttons.append(aiBtn, notAiBtn);
    card.append(buttons);

    let finalStatus = statusMessage || "";
    if (!finalStatus && cooldownMs > 0) {
      finalStatus = `Vote cooldown: ${Math.ceil(cooldownMs / 1000)}s`;
    }
    if (finalStatus) {
      card.append(createNode("div", "fit-status", finalStatus));
    }
  }

  async function loadAndRenderReel(reelId, options = {}) {
    const token = ++currentLoadToken;
    if (options.showLoading !== false) {
      renderLoading(reelId);
    }

    const state = await fetchReelState(reelId, {
      forceNetwork: Boolean(options.forceNetwork)
    });

    if (token !== currentLoadToken || reelId !== currentReelId) {
      return;
    }

    if (state.kind === "known") {
      renderKnown(reelId, state.counts, state.verdict);
      maybeAutoSkipKnownReel(reelId, state.counts, state.verdict);
      return;
    }

    if (state.kind === "unknown") {
      renderUnknown(reelId);
      return;
    }

    renderOffline(reelId);
  }

  async function handleKnownVote(reelId, label) {
    if (reelId !== currentReelId) {
      return;
    }
    if (getCooldownRemainingMs(reelId) > 0) {
      return;
    }

    startCooldown(reelId);
    scheduleCooldownRefresh(reelId);

    const cachedEntry = reelCache.get(reelId);
    const cachedCounts = cachedEntry?.counts || { aiCount: 0, notAiCount: 0 };
    const cachedVerdict = cachedEntry?.verdict || null;
    renderKnown(reelId, cachedCounts, cachedVerdict, "Submitting vote...", true);

    const sourceUrl = location.href;
    const voteResult = await backendRequest(
      "POST",
      `/reels/${encodeURIComponent(reelId)}/vote`,
      { label, source_url: sourceUrl }
    );

    if (reelId !== currentReelId) {
      return;
    }

    if (!voteResult.ok && voteResult.status === 0) {
      renderOffline(reelId);
      return;
    }

    const refreshed = await fetchReelState(reelId, { forceNetwork: true });
    if (reelId !== currentReelId) {
      return;
    }

    if (refreshed.kind === "known") {
      renderKnown(reelId, refreshed.counts, refreshed.verdict, "Vote recorded.");
    } else if (refreshed.kind === "unknown") {
      renderUnknown(reelId, "Reel is still unrated on backend.");
    } else {
      renderOffline(reelId);
    }
  }

  async function handleUnknownVote(reelId, label) {
    if (reelId !== currentReelId) {
      return;
    }
    if (getCooldownRemainingMs(reelId) > 0) {
      return;
    }

    startCooldown(reelId);
    scheduleCooldownRefresh(reelId);
    renderUnknown(reelId, "Submitting vote...", true);

    const sourceUrl = location.href;

    // Required order for unrated reels: /post -> /vote -> /reels/{id}
    const postResult = await backendRequest(
      "POST",
      `/reels/${encodeURIComponent(reelId)}/post`,
      { source_url: sourceUrl }
    );

    if (reelId !== currentReelId) {
      return;
    }
    if (!postResult.ok) {
      if (postResult.status === 0) {
        renderOffline(reelId);
      } else {
        renderUnknown(reelId, `Failed to enqueue reel (HTTP ${postResult.status}).`);
      }
      return;
    }

    const voteResult = await backendRequest(
      "POST",
      `/reels/${encodeURIComponent(reelId)}/vote`,
      { label, source_url: sourceUrl }
    );

    if (reelId !== currentReelId) {
      return;
    }
    if (!voteResult.ok && voteResult.status === 0) {
      renderOffline(reelId);
      return;
    }

    const refreshed = await fetchReelState(reelId, { forceNetwork: true });
    if (reelId !== currentReelId) {
      return;
    }

    if (refreshed.kind === "known") {
      renderKnown(reelId, refreshed.counts, refreshed.verdict, "Vote recorded.");
    } else if (refreshed.kind === "unknown") {
      renderUnknown(reelId, "Submitted. Reel is not indexed yet.");
    } else {
      renderOffline(reelId);
    }
  }

  function detectAndHandleReel(force) {
    if (!isReelContext()) {
      currentReelId = null;
      hideOverlay();
      return;
    }

    const reelId = getReelId();
    if (!reelId) {
      hideOverlay();
      return;
    }

    if (!force && reelId === currentReelId) {
      return;
    }

    currentReelId = reelId;
    loadAndRenderReel(reelId, { showLoading: true, forceNetwork: false });
  }

  function scheduleDetect(force) {
    if (force) {
      if (detectTimer) {
        clearTimeout(detectTimer);
      }
      detectTimer = null;
      detectAndHandleReel(true);
      return;
    }

    if (detectTimer) {
      return;
    }

    detectTimer = window.setTimeout(() => {
      detectTimer = null;
      detectAndHandleReel(false);
    }, 120);
  }

  function monitorLocationChanges() {
    if (historyPatched) {
      return;
    }
    historyPatched = true;

    const triggerLocationCheck = () => {
      if (location.href === lastHref) {
        return;
      }
      lastHref = location.href;
      scheduleDetect(true);
    };

    const patchMethod = (methodName) => {
      const original = history[methodName];
      if (typeof original !== "function") {
        return;
      }
      history[methodName] = function patchedHistoryMethod(...args) {
        const result = original.apply(this, args);
        triggerLocationCheck();
        return result;
      };
    };

    patchMethod("pushState");
    patchMethod("replaceState");

    window.addEventListener("popstate", triggerLocationCheck);
    window.addEventListener("hashchange", triggerLocationCheck);

    window.setInterval(triggerLocationCheck, 500);
  }

  function monitorMutations() {
    if (observerStarted) {
      return;
    }

    const start = () => {
      if (observerStarted) {
        return;
      }
      if (!document.documentElement) {
        window.setTimeout(start, 120);
        return;
      }

      const observer = new MutationObserver(() => {
        scheduleDetect(false);
      });

      observer.observe(document.documentElement, {
        childList: true,
        subtree: true
      });
      observerStarted = true;
    };

    start();
  }

  function init() {
    ensureOverlay();
    watchClientSettings();
    loadClientSettings().finally(() => {
      monitorLocationChanges();
      monitorMutations();
      scheduleDetect(true);
    });
  }

  init();
})();
