import os
import time
import json
import uuid
from typing import Optional, Literal, Dict, Any

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- Config ----------------
VALKEY_URL = os.getenv("VALKEY_URL")
STREAM_JOBS = os.getenv("STREAM_JOBS")

r = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

app = FastAPI(title="Siren's Call - Valkey API", version="0.2.0")

# For local dev + extension. Tighten later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Key helpers ("schema") ----------------
def k_reel(reel_id: str) -> str:
    return f"reel:{reel_id}"

# ---------------- Models ----------------
class InitVideoRequest(BaseModel):
    video_id: str = Field(..., description="Unique id (e.g., from TwelveLabs)")
    source_url: Optional[str] = Field(None, description="Where user saw the video (social URL)")

class VoteRequest(BaseModel):
    label: Literal["ai", "not_ai"]
    source_url: Optional[str] = None

class EnqueueVideoLinkRequest(BaseModel):
    video_url: str = Field(..., description="Video link to queue for ingestion/analysis")
    source_url: Optional[str] = None
    claim: Optional[str] = None

class VideoResponse(BaseModel):
    video_id: str
    meta: Dict[str, Any]
    votes: Dict[str, int]

# ---------------- Base record initializer ----------------
def ensure_base(reel_id: str, source_url: Optional[str] = None) -> None:
    base_data = {
        "ai": 0,
        "not_ai": 0,
        "created_ts": str(int(time.time() * 1000))
    }
    if source_url:
        base_data["source_url"] = source_url

    # Use JSON.SET to store the base data as a JSON object
    r.json().set(k_reel(reel_id), '$', base_data)

def read_reel(reel_id: str) -> dict:
    # Use JSON.GET to retrieve the JSON object
    reel_data = r.json().get(k_reel(reel_id))
    if not reel_data:
        raise HTTPException(status_code=404, detail="reel_id not found")
    return reel_data

# ---------------- Job enqueue helper ----------------
def enqueue_job(job_type: str, payload: dict) -> str:
    event = {
        "type": job_type,
        "payload": json.dumps(payload),
        "ts": str(int(time.time() * 1000)),
        "request_id": str(uuid.uuid4()),
    }
    msg_id = r.xadd(STREAM_JOBS, event)
    return msg_id

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "valkey": bool(r.ping())}

@app.post("/reels/init")
def init_reel(reel_id: str, source_url: Optional[str] = None):
    ensure_base(reel_id, source_url)
    return read_reel(reel_id)

@app.get("/reels/{reel_id}")
def get_reel(reel_id: str):
    return read_reel(reel_id)

@app.post("/reels/{reel_id}/vote")
def vote(reel_id: str, req: VoteRequest):
    # Enqueue vote job; worker will apply HINCRBY
    ensure_base(reel_id, req.source_url)

    msg_id = enqueue_job("vote", {
        "reel_id": reel_id,
        "label": req.label,
        "source_url": req.source_url or "",
    })
    return {"ok": True, "queued": True, "msg_id": msg_id}

@app.post("/jobs/reel_link")
def queue_reel_link(req: EnqueueVideoLinkRequest):
    msg_id = enqueue_job("reel_link", {
        "reel_url": req.video_url,
        "source_url": req.source_url or "",
        "claim": req.claim or "",
    })
    return {"ok": True, "queued": True, "msg_id": msg_id}

@app.get("/reels/{reel_id}/analysis")
def get_analysis(reel_id: str):
    raw = r.get(k_reel(reel_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return json.loads(raw)
