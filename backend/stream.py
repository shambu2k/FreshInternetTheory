import os
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")
STREAM_JOBS = os.getenv("STREAM_JOBS", os.getenv("WORKER_QUEUE_NAME", "reel_jobs"))
STREAM_GROUP = os.getenv("STREAM_GROUP", os.getenv("WORKER_STREAM_GROUP", "g_reel"))

r = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

# Creates and starts the configured stream/group used by backend and worker.
def create_and_start_reel_jobs_stream(stream_name: str = STREAM_JOBS, group_name: str = STREAM_GROUP):
    try:
        r.xgroup_create(stream_name, group_name, id="0-0", mkstream=True)
        print(f"✅ Created and started stream '{stream_name}' with group '{group_name}'")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"Group '{group_name}' already exists on '{stream_name}'")
        else:
            raise

if __name__ == "__main__":
    create_and_start_reel_jobs_stream()
