import os
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VALKEY_URL = os.getenv("VALKEY_URL", "redis://localhost:6379/0")

r = redis.Redis.from_url(VALKEY_URL, decode_responses=True)

# Refactored code to only create and start the stream named "reel_jobs" with group name "g_reel"
def create_and_start_reel_jobs_stream():
    try:
        r.xgroup_create("reel_jobs", "g_reel", id="0-0", mkstream=True)
        print("✅ Created and started stream 'reel_jobs' with group 'g_reel'")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print("Group 'g_reel' already exists on 'reel_jobs'")
        else:
            raise

if __name__ == "__main__":
    create_and_start_reel_jobs_stream()
