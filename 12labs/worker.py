#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import valkey
from valkey.exceptions import TimeoutError as ValkeyTimeoutError

from reel_downloader import download_facebook_reel, extract_reel_id
from upload import create_client, load_environment, upload_and_index_video

# Credits: https://github.com/aahilshaikh-twlbs/Recurser/blob/main/backend/app.py#L1023
ai_detection_categories = {
    "facial_artifacts": "unnatural facial symmetry, artificial facial proportions, synthetic facial structure, unnatural eye movements, artificial skin texture, robotic facial expressions",
    
    "motion_artifacts": "jerky movements, unnatural motion blur, artificial motion smoothing, synthetic frame transitions, mechanical object tracking, temporal inconsistencies",
    
    "lighting_artifacts": "inconsistent lighting, artificial shadow patterns, unnatural light sources, synthetic illumination, artificial ambient lighting",
    
    "audio_artifacts": "robotic speech patterns, artificial voice modulation, synthetic intonation, unnatural speech rhythm, artificial pronunciation",
    
    "environmental_artifacts": "inconsistent environmental details, artificial background elements, synthetic scene composition, unnatural object placement, impossible physics scenarios",
    
    "ai_generation_artifacts": "GAN artifacts, diffusion model artifacts, deep learning artifacts, machine learning artifacts, AI generation artifacts, artificial compression patterns",
    
    "behavioral_artifacts": "cat drinking tea, animals doing human activities, impossible animal behavior, unnatural animal interactions, synthetic animal movements, infants speaking fluently, babies doing adult tasks",
    
    "quality_artifacts": "inconsistent video quality, artificial quality patterns, synthetic quality variations",
    
    "texture_artifacts": "artificial texture patterns, synthetic material properties, unnatural surface details, artificial fabric textures, synthetic skin textures",
    
    "color_artifacts": "unnatural color saturation, artificial color grading, synthetic color palettes, unnatural color transitions, artificial color consistency",
    
    "perspective_artifacts": "impossible perspective angles, artificial depth perception, synthetic 3D rendering, unnatural camera angles, artificial spatial relationships",
    
    "temporal_artifacts": "unnatural time progression, artificial frame rates, synthetic temporal consistency, unnatural scene transitions, artificial pacing",
    
    "composition_artifacts": "artificial scene composition, synthetic framing, unnatural visual balance, artificial rule of thirds, synthetic visual hierarchy",
    
    "detail_artifacts": "artificial fine details, synthetic micro-movements, unnatural precision, artificial sharpness, synthetic clarity patterns",
    
    "interaction_artifacts": "unnatural object interactions, artificial physics, synthetic collision detection, unnatural gravity effects, artificial material responses",
    
    "thematic_artifacts": "anachronistic figures (e.g., Jesus, historical figures in modern settings), mythological events presented as real footage, hyper-religious engagement bait, staged miraculous feats, unlikely celebrity scenarios",

    "developmental_artifacts": "mismatch between biological age and capability, infants articulating complex sentences, toddlers driving or operating machinery, animals showing human-level intelligence",
    
    "aesthetic_artifacts": "hyper-realistic HDR look, excessive bloom or shine, 'plastic' skin on human figures, overly cinematic lighting in casual settings, dream-like or painterly composition style typical of Midjourney/Sora"
}
category_instructions = "\n".join([f"- {key}: {desc}" for key, desc in ai_detection_categories.items()])

DEFAULT_AI_DETECTION_PROMPT = f"""
Act as a forensic video analyst. Analyze this video for AI generation.
Classify evidence into these specific categories:

{category_instructions}

Return JSON with keys: is_likely_ai_generated (bool), confidence (0-100), key_evidence (list of objects with category, timestamp, description).
""".strip()


@dataclass(frozen=True)
class WorkerConfig:
    valkey_url: str
    queue_name: str
    queue_type: str
    stream_group: str
    stream_consumer: str
    stream_block_ms: int
    socket_timeout_seconds: float
    poll_interval_seconds: int
    downloads_dir: Path
    analysis_prompt: str
    datastore_key_prefix: str


def load_worker_config() -> WorkerConfig:
    load_environment()
    valkey_url = os.getenv("WORKER_VALKEY_URL", "valkey://localhost:6379/0")
    queue_name = os.getenv("WORKER_QUEUE_NAME", "reel_jobs")
    queue_type = os.getenv("WORKER_QUEUE_TYPE", "stream")
    stream_group = os.getenv("WORKER_STREAM_GROUP", "g_reel")
    stream_consumer = os.getenv("WORKER_STREAM_CONSUMER", "worker-1")
    stream_block_ms = int(os.getenv("WORKER_STREAM_BLOCK_MS", "5000"))
    default_socket_timeout = max(30.0, (stream_block_ms / 1000.0) + 10.0)
    socket_timeout_seconds = float(os.getenv("WORKER_VALKEY_SOCKET_TIMEOUT_SECONDS", str(default_socket_timeout)))
    poll_interval = int(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "5"))
    downloads_dir = Path(os.getenv("WORKER_DOWNLOADS_DIR", Path(__file__).resolve().parent / "downloads"))
    prompt = os.getenv("WORKER_AI_DETECTION_PROMPT", DEFAULT_AI_DETECTION_PROMPT)
    datastore_key_prefix = os.getenv("WORKER_DATASTORE_KEY_PREFIX", "")
    return WorkerConfig(
        valkey_url=valkey_url,
        queue_name=queue_name,
        queue_type=queue_type,
        stream_group=stream_group,
        stream_consumer=stream_consumer,
        stream_block_ms=stream_block_ms,
        socket_timeout_seconds=socket_timeout_seconds,
        poll_interval_seconds=poll_interval,
        downloads_dir=downloads_dir,
        analysis_prompt=prompt,
        datastore_key_prefix=datastore_key_prefix,
    )


def parse_queue_message(raw_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid queue payload JSON: {exc}") from exc

    reel_url = payload.get("reel_url")
    if not reel_url:
        raise ValueError("Queue payload must include 'reel_url'")
    return payload


def parse_stream_fields(fields: dict[str, Any]) -> dict[str, Any]:
    if "payload" in fields:
        return parse_queue_message(fields["payload"])

    reel_url = fields.get("reel_url")
    if not reel_url:
        raise ValueError("Stream message must include 'reel_url' or 'payload'")

    payload: dict[str, Any] = {"reel_url": reel_url}
    if "job_id" in fields:
        payload["job_id"] = fields["job_id"]
    return payload


def analyze_video_for_ai(client: Any, video_id: str, prompt: str) -> dict[str, Any]:
    response = client.analyze(video_id=video_id, prompt=prompt)
    response_data = getattr(response, "data", response)

    if isinstance(response_data, dict):
        return response_data

    if isinstance(response_data, str):
        try:
            return json.loads(response_data)
        except json.JSONDecodeError:
            return {"raw_analysis": response_data}

    return {"raw_analysis": str(response_data)}


def save_result_to_datastore(
    datastore_client: valkey.Valkey,
    result: dict[str, Any],
    key_prefix: str = "",
) -> None:
    reel_id = result.get("reel_id")
    analysis = result.get("analysis")
    if not reel_id:
        raise ValueError("Cannot persist result without reel_id")

    datastore_key = f"{key_prefix}{reel_id}"
    datastore_value = json.dumps(analysis, ensure_ascii=False)
    datastore_client.set(datastore_key, datastore_value)
    print(f"Saved analysis to datastore key={datastore_key}")


def process_message(
    client: Any,
    config: WorkerConfig,
    payload: dict[str, Any],
    datastore_client: valkey.Valkey,
) -> dict[str, Any]:
    reel_url = payload["reel_url"]
    job_id = payload.get("job_id")

    downloaded_path = download_facebook_reel(reel_url, output_dir=config.downloads_dir)
    indexing_result = upload_and_index_video(downloaded_path)
    analysis_result = analyze_video_for_ai(
        client=client,
        video_id=indexing_result.indexed_asset_id,
        prompt=config.analysis_prompt,
    )

    result = {
        "job_id": job_id,
        "reel_url": reel_url,
        "reel_id": extract_reel_id(reel_url),
        "downloaded_file": str(downloaded_path),
        "indexing": asdict(indexing_result),
        "analysis": analysis_result,
        "processed_at_epoch": int(time.time()),
    }

    save_result_to_datastore(
        datastore_client=datastore_client,
        result=result,
        key_prefix=config.datastore_key_prefix,
    )
    return result


def run_worker() -> None:
    config = load_worker_config()
    valkey_client = valkey.from_url(
        config.valkey_url,
        decode_responses=True,
        socket_timeout=config.socket_timeout_seconds,
    )
    twelvelabs_client = create_client()

    print(f"Worker listening on {config.queue_type} '{config.queue_name}' at {config.valkey_url}")

    if config.queue_type == "stream":
        print(
            f"Waiting for stream='{config.queue_name}' group='{config.stream_group}' to be available..."
        )

    while True:
        if config.queue_type == "stream":
            try:
                entries = valkey_client.xreadgroup(
                    groupname=config.stream_group,
                    consumername=config.stream_consumer,
                    streams={config.queue_name: ">"},
                    count=1,
                    block=config.stream_block_ms,
                )
            except ValkeyTimeoutError:
                continue
            except valkey.exceptions.ResponseError as exc:
                error_text = str(exc).lower()
                if "nogroup" in error_text or "no such key" in error_text:
                    print(
                        f"Stream/group not ready yet (stream='{config.queue_name}', group='{config.stream_group}'). Retrying..."
                    )
                    time.sleep(config.poll_interval_seconds)
                    continue
                raise
            if not entries:
                continue

            for stream_name, messages in entries:
                for message_id, fields in messages:
                    try:
                        payload = parse_stream_fields(fields)
                        print(f"Received stream job for reel URL: {payload['reel_url']}")
                        result = process_message(
                            twelvelabs_client,
                            config,
                            payload,
                            datastore_client=valkey_client,
                        )
                        valkey_client.xack(stream_name, config.stream_group, message_id)
                        print(json.dumps({"status": "ok", "result": result, "stream_id": message_id}, ensure_ascii=False))
                    except Exception as exc:
                        print(json.dumps({"status": "error", "message": str(exc), "stream_id": message_id, "fields": fields}, ensure_ascii=False))
        else:
            item = valkey_client.blpop(config.queue_name, timeout=0)
            if not item:
                continue

            _, raw_message = item
            try:
                payload = parse_queue_message(raw_message)
                print(f"Received job: {payload.get('job_id')} for reel URL: {payload['reel_url']}")
                result = process_message(
                    twelvelabs_client,
                    config,
                    payload,
                    datastore_client=valkey_client,
                )
                print(json.dumps({"status": "ok", "result": result}, ensure_ascii=False))
            except Exception as exc:
                print(json.dumps({"status": "error", "message": str(exc), "payload": raw_message}, ensure_ascii=False))


if __name__ == "__main__":
    run_worker()
