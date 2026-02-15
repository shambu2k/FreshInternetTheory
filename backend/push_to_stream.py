import redis
import json
import os
import time
import uuid

# Redis connection setup
VALKEY_URL = os.getenv("VALKEY_URL", "redis://127.0.0.1:6379")
STREAM_NAME = "reel_jobs"

r = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

def push_to_stream():
    payload = {
        "reel_url": "https://www.facebook.com/reel/1441339397657413",
        "ts": str(int(time.time() * 1000)),
        "request_id": str(uuid.uuid4()),
    }
    msg_id = r.xadd(STREAM_NAME, payload)
    print(f"✅ Pushed payload to stream '{STREAM_NAME}' with ID: {msg_id}")

if __name__ == "__main__":
    push_to_stream()