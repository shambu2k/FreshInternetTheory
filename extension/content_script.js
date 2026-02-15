(() => {
  const CACHE_TTL_MS = {
    known: 30 * 1000,
    unknown: 2 * 60 * 1000
  };
  const VOTE_COOLDOWN_MS = 10 * 1000;
  const OVERLAY_HOST_ID = "fit-reel-overlay-host";

  const reelCache = new Map();
  const voteCooldowns = new Map();

  let overlayHost = null;
  let overlayContent = null;
  let currentReelId = null;
  let lastHref = location.href;
  let currentLoadToken = 0;
  let observerStarted = false;
  let historyPatched = false;
  let detectTimer = null;

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
      overlayHost.style.right = "16px";
      overlayHost.style.zIndex = "2147483647";
      overlayHost.style.maxWidth = "300px";
      overlayHost.style.pointerEvents = "none";
      document.documentElement.appendChild(overlayHost);
    }

    if (!overlayHost.shadowRoot) {
      const shadow = overlayHost.attachShadow({ mode: "open" });

      const style = document.createElement("style");
      style.textContent = `
        .fit-card {
          pointer-events: auto;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          width: 280px;
          background: rgba(20, 20, 20, 0.92);
          color: #ffffff;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.18);
          box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
          padding: 12px;
        }
        .fit-title {
          font-size: 14px;
          font-weight: 700;
          margin-bottom: 6px;
        }
        .fit-subtitle {
          font-size: 12px;
          opacity: 0.75;
          margin-bottom: 10px;
          line-height: 1.4;
          word-break: break-all;
        }
        .fit-stat {
          font-size: 13px;
          margin-bottom: 4px;
          line-height: 1.3;
        }
        .fit-question {
          margin-top: 8px;
          font-size: 13px;
          line-height: 1.35;
        }
        .fit-buttons {
          margin-top: 10px;
          display: flex;
          gap: 8px;
        }
        .fit-btn {
          flex: 1;
          border: none;
          border-radius: 8px;
          padding: 8px 10px;
          font-size: 12px;
          font-weight: 700;
          cursor: pointer;
          transition: opacity 120ms ease;
        }
        .fit-btn:hover:not(:disabled) {
          opacity: 0.88;
        }
        .fit-btn:disabled {
          cursor: not-allowed;
          opacity: 0.55;
        }
        .fit-btn-ai {
          background: #00a86b;
          color: #fff;
        }
        .fit-btn-not-ai {
          background: #1f8fff;
          color: #fff;
        }
        .fit-status {
          margin-top: 8px;
          font-size: 12px;
          opacity: 0.82;
          line-height: 1.35;
        }
        .fit-verdict {
          margin-bottom: 10px;
        }
        .fit-badge {
          display: inline-block;
          border-radius: 999px;
          padding: 5px 10px;
          font-size: 11px;
          font-weight: 800;
          letter-spacing: 0.2px;
          border: 1px solid transparent;
        }
        .fit-badge-ai {
          background: rgba(247, 77, 77, 0.18);
          border-color: rgba(255, 121, 121, 0.45);
          color: #ffd7d7;
        }
        .fit-badge-not-ai {
          background: rgba(67, 199, 129, 0.18);
          border-color: rgba(95, 227, 157, 0.45);
          color: #d8ffe8;
        }
        .fit-badge-pending {
          background: rgba(255, 198, 66, 0.18);
          border-color: rgba(255, 211, 106, 0.5);
          color: #ffecc3;
        }
        .fit-badge-failed {
          background: rgba(255, 119, 66, 0.2);
          border-color: rgba(255, 145, 102, 0.5);
          color: #ffe1d1;
        }
        .fit-badge-neutral {
          background: rgba(160, 167, 178, 0.2);
          border-color: rgba(190, 197, 210, 0.5);
          color: #edf0f7;
        }
        .fit-accordion {
          margin-top: 8px;
          border: 1px solid rgba(255, 255, 255, 0.2);
          border-radius: 8px;
          padding: 2px 8px 7px;
          background: rgba(255, 255, 255, 0.04);
        }
        .fit-accordion summary {
          cursor: pointer;
          font-size: 12px;
          font-weight: 700;
          padding: 5px 0;
          list-style: none;
          outline: none;
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
          opacity: 0.95;
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

    const title = createNode("div", "fit-title", "Reel AI Rating");
    const subtitle = createNode("div", "fit-subtitle", `reel_id: ${reelId}`);

    overlayContent.append(title, subtitle);
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
          badgeText: `Backend: Likely AI${confidenceText}`,
          reasonLines
        };
      }

      if (isLikelyAi === false) {
        return {
          tone: "not-ai",
          badgeText: `Backend: Likely Not AI${confidenceText}`,
          reasonLines
        };
      }

      return {
        tone: "neutral",
        badgeText: `Backend: Analysis available${confidenceText}`,
        reasonLines
      };
    }

    if (status === "queued") {
      return {
        tone: "pending",
        badgeText: "Backend: Analysis queued",
        reasonLines: []
      };
    }

    if (status === "not_started") {
      return {
        tone: "pending",
        badgeText: "Backend: Analysis not started",
        reasonLines: []
      };
    }

    if (status === "failed") {
      return {
        tone: "failed",
        badgeText: "Backend: Analysis failed",
        reasonLines: extractReasonLines(null, analysis.last_error || "No error details available.")
      };
    }

    return {
      tone: "neutral",
      badgeText: "Backend: No analysis yet",
      reasonLines: []
    };
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
    monitorLocationChanges();
    monitorMutations();
    scheduleDetect(true);
  }

  init();
})();
