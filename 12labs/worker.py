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

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class WorkerConfig:
    valkey_url: str
    queue_name: str
    queue_type: str
    stream_group: str
    stream_dead_letter_name: str
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
    stream_dead_letter_name = os.getenv("WORKER_STREAM_DLQ_NAME", f"{queue_name}_dlq")
    stream_consumer = os.getenv("WORKER_STREAM_CONSUMER", "worker-1")
    stream_block_ms = int(os.getenv("WORKER_STREAM_BLOCK_MS", "5000"))
    default_socket_timeout = max(30.0, (stream_block_ms / 1000.0) + 10.0)
    socket_timeout_seconds = float(os.getenv("WORKER_VALKEY_SOCKET_TIMEOUT_SECONDS", str(default_socket_timeout)))
    poll_interval = int(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "5"))
    downloads_dir = Path(os.getenv("WORKER_DOWNLOADS_DIR", Path(__file__).resolve().parent / "downloads"))
    prompt = os.getenv("WORKER_AI_DETECTION_PROMPT", DEFAULT_AI_DETECTION_PROMPT)
    datastore_key_prefix = os.getenv("WORKER_DATASTORE_KEY_PREFIX", "reel:")
    return WorkerConfig(
        valkey_url=valkey_url,
        queue_name=queue_name,
        queue_type=queue_type,
        stream_group=stream_group,
        stream_dead_letter_name=stream_dead_letter_name,
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
    elif "request_id" in fields:
        payload["job_id"] = fields["request_id"]
    if "reel_id" in fields:
        payload["reel_id"] = fields["reel_id"]
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


def now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def build_default_reel_document(reel_id: str, source_url: str | None = None) -> dict[str, Any]:
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
    source_url: str | None = None,
) -> dict[str, Any]:
    doc = raw_doc.copy() if isinstance(raw_doc, dict) else {}
    defaults = build_default_reel_document(reel_id, source_url=source_url)

    doc["schema_version"] = SCHEMA_VERSION
    doc["reel_id"] = reel_id
    doc["source_url"] = source_url if source_url else doc.get("source_url")
    doc["post_count"] = _coerce_non_negative_int(doc.get("post_count"), 0)

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


def save_result_to_datastore(
    datastore_client: valkey.Valkey,
    result: dict[str, Any],
    key_prefix: str = "",
) -> None:
    reel_id = result.get("reel_id")
    analysis = result.get("analysis")
    indexing = result.get("indexing")
    processed_at_epoch = result.get("processed_at_epoch")
    reel_url = result.get("reel_url")
    job_id = result.get("job_id")
    if not reel_id:
        raise ValueError("Cannot persist result without reel_id")

    datastore_key = f"{key_prefix}{reel_id}"
    current_value = datastore_client.get(datastore_key)
    if not current_value and key_prefix:
        # Backward compatibility for older worker versions that used the raw reel_id key.
        current_value = datastore_client.get(reel_id)

    if current_value:
        try:
            current_dict = json.loads(current_value)
        except json.JSONDecodeError:
            current_dict = {}
    else:
        current_dict = {}

    current_dict = normalize_reel_document(
        reel_id=reel_id,
        raw_doc=current_dict,
        source_url=reel_url,
    )

    event_ts = now_ms()
    current_dict["analysis"]["status"] = "completed"
    current_dict["analysis"]["twelvelabs"] = analysis
    current_dict["analysis"]["indexing"] = indexing
    if job_id:
        current_dict["analysis"]["last_job_id"] = job_id
    current_dict["analysis"]["last_error"] = None
    current_dict["analysis"]["processed_at_epoch"] = processed_at_epoch
    current_dict["analysis"]["updated_at_ms"] = event_ts
    current_dict["timestamps"]["last_analyzed_at_ms"] = event_ts
    current_dict["timestamps"]["updated_at_ms"] = event_ts

    # Legacy alias for existing readers.
    current_dict["12labs_ai_analysis"] = analysis

    datastore_client.set(datastore_key, json.dumps(current_dict, ensure_ascii=False))
    print(f"Saved analysis to datastore key={datastore_key}")


def process_message(
    client: Any,
    config: WorkerConfig,
    payload: dict[str, Any],
    datastore_client: valkey.Valkey,
) -> dict[str, Any]:
    reel_url = payload["reel_url"]
    job_id = payload.get("job_id")
    reel_id = payload.get("reel_id")
    if not reel_id:
        reel_id = extract_reel_id(reel_url)

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
        "reel_id": reel_id,
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
                        print(
                            json.dumps(
                                {
                                    "status": "error",
                                    "message": str(exc),
                                    "stream_id": message_id,
                                    "fields": fields,
                                },
                                ensure_ascii=False,
                            )
                        )
                        dlq_event = {
                            "stream_id": message_id,
                            "error": str(exc),
                            "fields": json.dumps(fields, ensure_ascii=False),
                            "ts": str(now_ms()),
                        }
                        try:
                            valkey_client.xadd(config.stream_dead_letter_name, dlq_event)
                        except Exception as dlq_exc:
                            print(
                                json.dumps(
                                    {
                                        "status": "error",
                                        "message": f"Failed to write to DLQ: {dlq_exc}",
                                        "stream_id": message_id,
                                        "dlq_stream": config.stream_dead_letter_name,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        valkey_client.xack(stream_name, config.stream_group, message_id)
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
