import json
import os
import re
import time
import uuid
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://127.0.0.1:6379")
STREAM_JOBS = os.getenv("STREAM_JOBS", "reel_jobs")
SCHEMA_VERSION = 2

r = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

app = FastAPI(title="Siren's Call - Valkey API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_ms() -> int:
    return int(time.time() * 1000)


def k_reel(reel_id: str) -> str:
    return f"reel:{reel_id}"


def k_reel_count(reel_id: str) -> str:
    # Legacy key. Migrated into reel:{reel_id}.post_count.
    return f"reel_count:{reel_id}"


def extract_reel_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    match = re.search(r"/reels?/(\d+)", parsed.path.rstrip("/"))
    if not match:
        raise ValueError(
            "Could not extract reel_id from source_url. Expected format like "
            "https://www.facebook.com/reel/1031867656387604"
        )
    return match.group(1)


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def build_default_reel_document(reel_id: str, source_url: Optional[str] = None) -> dict[str, Any]:
    ts = now_ms()
    return {
        "schema_version": SCHEMA_VERSION,
        "reel_id": reel_id,
        "source_url": source_url,
        "post_count": 0,
        "votes": {"ai": 0, "not_ai": 0, "total": 0},
        "user_votes": [],
        "analysis": {
            "status": "not_started",
            "twelvelabs": None,
            "indexing": None,
            "embedding_signature": None,
            "similarity_match": None,
            "last_job_id": None,
            "last_error": None,
            "processed_at_epoch": None,
            "updated_at_ms": None,
        },
        "timestamps": {
            "created_at_ms": ts,
            "updated_at_ms": ts,
            "last_posted_at_ms": None,
            "last_voted_at_ms": None,
            "last_analyzed_at_ms": None,
        },
        # Legacy aliases kept for compatibility.
        "12labs_ai_analysis": None,
        "user_responses": [],
        "user_analysis": {"votes": [], "counts": {"ai": 0, "not_ai": 0, "total": 0}},
    }


def normalize_reel_document(
    reel_id: str,
    raw_doc: Any,
    *,
    source_url: Optional[str] = None,
    legacy_post_count: int = 0,
) -> dict[str, Any]:
    doc = raw_doc.copy() if isinstance(raw_doc, dict) else {}
    defaults = build_default_reel_document(reel_id, source_url=source_url)

    doc["schema_version"] = SCHEMA_VERSION
    doc["reel_id"] = reel_id
    doc["source_url"] = source_url if source_url else doc.get("source_url")

    doc["post_count"] = max(
        _coerce_non_negative_int(doc.get("post_count"), 0),
        _coerce_non_negative_int(legacy_post_count, 0),
    )

    user_votes_raw = doc.get("user_votes")
    if not isinstance(user_votes_raw, list):
        user_votes_raw = doc.get("user_responses", [])
    if not isinstance(user_votes_raw, list):
        user_votes_raw = []

    normalized_votes: list[dict[str, Any]] = []
    for item in user_votes_raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if label not in ("ai", "not_ai"):
            continue
        vote_timestamp = _coerce_non_negative_int(
            item.get("timestamp_ms", item.get("timestamp")),
            now_ms(),
        )
        vote_source_url = item.get("source_url")
        normalized_vote: dict[str, Any] = {"label": label, "timestamp_ms": vote_timestamp}
        if isinstance(vote_source_url, str) and vote_source_url:
            normalized_vote["source_url"] = vote_source_url
        normalized_votes.append(normalized_vote)

    votes = doc.get("votes")
    if not isinstance(votes, dict):
        votes = {}
    inferred_ai_votes = sum(1 for vote in normalized_votes if vote["label"] == "ai")
    inferred_not_ai_votes = sum(1 for vote in normalized_votes if vote["label"] == "not_ai")
    ai_votes = max(_coerce_non_negative_int(votes.get("ai"), 0), inferred_ai_votes)
    not_ai_votes = max(_coerce_non_negative_int(votes.get("not_ai"), 0), inferred_not_ai_votes)
    total_votes = max(_coerce_non_negative_int(votes.get("total"), 0), ai_votes + not_ai_votes)

    analysis = doc.get("analysis")
    if not isinstance(analysis, dict):
        analysis = {}
    if "twelvelabs" not in analysis and "12labs_ai_analysis" in doc:
        analysis["twelvelabs"] = doc.get("12labs_ai_analysis")
    status = analysis.get("status")
    if status not in {"not_started", "queued", "completed", "failed"}:
        status = "completed" if analysis.get("twelvelabs") else "not_started"
    analysis = {
        "status": status,
        "twelvelabs": analysis.get("twelvelabs"),
        "indexing": analysis.get("indexing"),
        "embedding_signature": analysis.get("embedding_signature"),
        "similarity_match": analysis.get("similarity_match"),
        "last_job_id": analysis.get("last_job_id"),
        "last_error": analysis.get("last_error"),
        "processed_at_epoch": analysis.get("processed_at_epoch"),
        "updated_at_ms": analysis.get("updated_at_ms"),
    }

    timestamps = doc.get("timestamps")
    if not isinstance(timestamps, dict):
        timestamps = {}
    created_at_ms = _coerce_non_negative_int(
        timestamps.get("created_at_ms"),
        defaults["timestamps"]["created_at_ms"],
    )
    timestamps = {
        "created_at_ms": created_at_ms,
        "updated_at_ms": _coerce_non_negative_int(
            timestamps.get("updated_at_ms"),
            created_at_ms,
        ),
        "last_posted_at_ms": timestamps.get("last_posted_at_ms"),
        "last_voted_at_ms": timestamps.get("last_voted_at_ms"),
        "last_analyzed_at_ms": timestamps.get("last_analyzed_at_ms"),
    }

    doc["votes"] = {"ai": ai_votes, "not_ai": not_ai_votes, "total": total_votes}
    doc["user_votes"] = normalized_votes
    doc["analysis"] = analysis
    doc["timestamps"] = timestamps

    # Legacy aliases kept in sync with canonical fields.
    doc["12labs_ai_analysis"] = analysis["twelvelabs"]
    doc["user_responses"] = normalized_votes
    doc["user_analysis"] = {"votes": normalized_votes, "counts": doc["votes"]}

    return doc


def _load_raw_reel_json(reel_id: str) -> Optional[str]:
    raw = r.get(k_reel(reel_id))
    if raw is not None:
        return raw

    # Legacy fallback from worker's old key format.
    legacy_raw = r.get(reel_id)
    if legacy_raw is not None:
        r.set(k_reel(reel_id), legacy_raw)
    return legacy_raw


def read_reel_optional(reel_id: str) -> Optional[dict[str, Any]]:
    raw = _load_raw_reel_json(reel_id)
    if raw is None:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt reel record for {reel_id}: {exc}") from exc

    legacy_post_count = _coerce_non_negative_int(r.get(k_reel_count(reel_id)), 0)
    return normalize_reel_document(
        reel_id=reel_id,
        raw_doc=parsed,
        legacy_post_count=legacy_post_count,
    )


def read_reel(reel_id: str) -> dict[str, Any]:
    record = read_reel_optional(reel_id)
    if record is None:
        raise HTTPException(status_code=404, detail="reel_id not found")
    return record


def persist_reel(reel_id: str, reel_data: dict[str, Any]) -> None:
    normalized = normalize_reel_document(reel_id, reel_data)
    normalized["timestamps"]["updated_at_ms"] = now_ms()
    r.set(k_reel(reel_id), json.dumps(normalized, ensure_ascii=False))


def enqueue_job(reel_url: str, reel_id: str, job_id: str) -> str:
    event = {
        "reel_url": reel_url,
        "reel_id": reel_id,
        "job_id": job_id,
        "request_id": job_id,  # backward compatibility for existing consumers
        "ts": str(now_ms()),
    }
    return r.xadd(STREAM_JOBS, event)


def validate_reel_url_for_path(reel_id: str, reel_url: str) -> None:
    try:
        extracted_reel_id = extract_reel_id(reel_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if extracted_reel_id != reel_id:
        raise HTTPException(
            status_code=400,
            detail=f"reel_id path ({reel_id}) does not match source_url reel_id ({extracted_reel_id})",
        )


class VoteRequest(BaseModel):
    label: Literal["ai", "not_ai"]
    source_url: Optional[str] = None


class EnqueueVideoLinkRequest(BaseModel):
    source_url: Optional[str] = None


@app.get("/health")
def health() -> dict[str, bool]:
    try:
        is_healthy = bool(r.ping())
    except redis.RedisError:
        is_healthy = False
    return {"ok": is_healthy, "valkey": is_healthy}


@app.get("/reels/{reel_id}")
def get_reel(reel_id: str) -> dict[str, Any]:
    reel_data = read_reel(reel_id)
    return {
        "reel_data": reel_data,
        "post_count": reel_data["post_count"],
        # Backward compatibility for older consumers expecting request_count.
        "request_count": reel_data["post_count"],
    }


@app.post("/reels/{reel_id}/post")
def post_reel(reel_id: str, req: EnqueueVideoLinkRequest) -> dict[str, Any]:
    existing = read_reel_optional(reel_id)
    reel_url = req.source_url or (existing or {}).get("source_url")
    if not reel_url:
        raise HTTPException(
            status_code=400,
            detail="source_url is required for a new reel record",
        )
    validate_reel_url_for_path(reel_id, reel_url)

    job_id = str(uuid.uuid4())
    msg_id = enqueue_job(reel_url=reel_url, reel_id=reel_id, job_id=job_id)

    reel_data = normalize_reel_document(
        reel_id=reel_id,
        raw_doc=existing,
        source_url=reel_url,
        legacy_post_count=_coerce_non_negative_int(r.get(k_reel_count(reel_id)), 0),
    )
    event_ts = now_ms()
    reel_data["post_count"] += 1
    reel_data["analysis"]["status"] = "queued"
    reel_data["analysis"]["last_job_id"] = job_id
    reel_data["analysis"]["last_error"] = None
    reel_data["analysis"]["updated_at_ms"] = event_ts
    reel_data["timestamps"]["last_posted_at_ms"] = event_ts
    persist_reel(reel_id, reel_data)

    # Keep old counter key in sync until all clients migrate.
    r.set(k_reel_count(reel_id), reel_data["post_count"])

    return {
        "ok": True,
        "queued": True,
        "job_id": job_id,
        "msg_id": msg_id,
        "reel_id": reel_id,
        "post_count": reel_data["post_count"],
    }


@app.post("/reels/{reel_id}/vote")
def vote_reel(reel_id: str, req: VoteRequest) -> dict[str, Any]:
    if req.source_url:
        validate_reel_url_for_path(reel_id, req.source_url)

    existing = read_reel_optional(reel_id)
    reel_data = normalize_reel_document(
        reel_id=reel_id,
        raw_doc=existing,
        source_url=req.source_url or (existing or {}).get("source_url"),
        legacy_post_count=_coerce_non_negative_int(r.get(k_reel_count(reel_id)), 0),
    )

    vote_ts = now_ms()
    vote_entry: dict[str, Any] = {"label": req.label, "timestamp_ms": vote_ts}
    if req.source_url:
        vote_entry["source_url"] = req.source_url
    reel_data["user_votes"].append(vote_entry)
    reel_data["votes"][req.label] += 1
    reel_data["votes"]["total"] = reel_data["votes"]["ai"] + reel_data["votes"]["not_ai"]
    reel_data["timestamps"]["last_voted_at_ms"] = vote_ts
    persist_reel(reel_id, reel_data)

    return {
        "ok": True,
        "message": "User response recorded.",
        "votes": reel_data["votes"],
        "total_votes": reel_data["votes"]["total"],
    }
