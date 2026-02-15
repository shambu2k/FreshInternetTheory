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
VALKEY_URL = os.getenv("VALKEY_URL", "redis://127.0.0.1:6379")
STREAM_JOBS = os.getenv("STREAM_JOBS", "reel_jobs")

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

# ---------------- Key helpers (Updated) ----------------
def k_reel_count(reel_id: str) -> str:
    return f"reel_count:{reel_id}"

# ---------------- Models ----------------
class InitVideoRequest(BaseModel):
    video_id: str = Field(..., description="Unique id (e.g., from TwelveLabs)")
    source_url: Optional[str] = Field(None, description="Where user saw the video (social URL)")

class VoteRequest(BaseModel):
    label: Literal["ai", "not_ai"]
    source_url: Optional[str] = None

class EnqueueVideoLinkRequest(BaseModel):
    source_url: Optional[str] = None

class VideoResponse(BaseModel):
    video_id: str
    meta: Dict[str, Any]
    votes: Dict[str, int]

def read_reel(reel_id: str) -> dict:
    # Use JSON.GET to retrieve the JSON object
    reel_data = r.json().get(k_reel(reel_id))
    if not reel_data:
        raise HTTPException(status_code=404, detail="reel_id not found")
    return reel_data

# ---------------- Job enqueue helper ----------------
def enqueue_job(job_type: str, reel_url: str) -> str:
    event = {
        "reel_url": reel_url,
        "ts": str(int(time.time() * 1000)),
        "request_id": str(uuid.uuid4()),
    }
    msg_id = r.xadd(STREAM_JOBS, event)
    return msg_id

# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True, "valkey": bool(r.ping())}

@app.get("/reels/{reel_id}")
def get_reel(reel_id: str):
    reel_data = read_reel(reel_id)
    count_key = k_reel_count(reel_id)
    count = r.get(count_key) or 0
    return {"reel_data": reel_data, "request_count": int(count)}

@app.post("/reels/{reel_id}/post")
def post_reel(reel_id: str, req: EnqueueVideoLinkRequest):
    # Increment the request count for the reel_id
    count_key = k_reel_count(reel_id)
    print(f"Incrementing count for {reel_id} at key {count_key}")
    r.incr(count_key)

    # Enqueue the job
    msg_id = enqueue_job("reel_link", req.source_url)
    return {"ok": True, "queued": True, "msg_id": msg_id}

# ---------------- Updated Routes ----------------
@app.post("/reels/{reel_id}/vote")
def vote_reel(reel_id: str, req: VoteRequest):
    # Retrieve the existing reel data
    reel_data = read_reel(reel_id)

    # Add the user's response to the user_responses column
    user_responses = reel_data.get("user_responses", [])
    user_responses.append({
        "label": req.label,
        "timestamp": str(int(time.time() * 1000))
    })

    # Update the reel data with the new user response
    reel_data["user_responses"] = user_responses
    r.json().set(k_reel(reel_id), '$', reel_data)

    return {"ok": True, "message": "User response recorded."}
