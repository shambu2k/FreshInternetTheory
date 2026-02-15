#!/usr/bin/env python3

from __future__ import annotations

import json
import inspect
import math
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
    similarity_reuse_enabled: bool
    similarity_reuse_threshold: float
    similarity_candidate_limit: int


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
    similarity_reuse_enabled = os.getenv("WORKER_SIMILARITY_REUSE_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    similarity_reuse_threshold = float(os.getenv("WORKER_SIMILARITY_REUSE_THRESHOLD", "0.92"))
    similarity_reuse_threshold = max(0.0, min(1.0, similarity_reuse_threshold))
    similarity_candidate_limit = int(os.getenv("WORKER_SIMILARITY_CANDIDATE_LIMIT", "1000"))
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
        similarity_reuse_enabled=similarity_reuse_enabled,
        similarity_reuse_threshold=similarity_reuse_threshold,
        similarity_candidate_limit=similarity_candidate_limit,
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


def _coerce_float_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    vector: list[float] = []
    for element in value:
        if isinstance(element, (int, float)):
            vector.append(float(element))
            continue
        return None
    return vector


def _extract_segment_embedding_vector(segment: Any) -> list[float] | None:
    # Supports both object-based and dict-based SDK response shapes.
    if isinstance(segment, dict):
        candidates = (
            segment.get("embeddings_float"),
            segment.get("embedding_float"),
            segment.get("embedding"),
            segment.get("vector"),
            segment.get("embeddings"),
        )
    else:
        candidates = (
            getattr(segment, "embeddings_float", None),
            getattr(segment, "embedding_float", None),
            getattr(segment, "embedding", None),
            getattr(segment, "vector", None),
            getattr(segment, "embeddings", None),
        )

    for candidate in candidates:
        if isinstance(candidate, dict):
            for nested_key in ("float", "embeddings_float", "vector", "values"):
                nested_value = candidate.get(nested_key)
                vector = _coerce_float_vector(nested_value)
                if vector:
                    return vector
        vector = _coerce_float_vector(candidate)
        if vector:
            return vector
    return None


def _extract_video_embedding_segments(indexed_asset_details: Any) -> list[Any]:
    # Newer SDK shape.
    video_embedding = getattr(indexed_asset_details, "video_embedding", None)
    if video_embedding is None and isinstance(indexed_asset_details, dict):
        video_embedding = indexed_asset_details.get("video_embedding")
    if video_embedding is not None:
        segments = getattr(video_embedding, "segments", None)
        if segments is None and isinstance(video_embedding, dict):
            segments = video_embedding.get("segments")
        if isinstance(segments, list):
            return segments

    # Older SDK shape.
    embedding = getattr(indexed_asset_details, "embedding", None)
    if embedding is None and isinstance(indexed_asset_details, dict):
        embedding = indexed_asset_details.get("embedding")
    if embedding is not None:
        nested_video_embedding = getattr(embedding, "video_embedding", None)
        if nested_video_embedding is None and isinstance(embedding, dict):
            nested_video_embedding = embedding.get("video_embedding")
        if nested_video_embedding is not None:
            segments = getattr(nested_video_embedding, "segments", None)
            if segments is None and isinstance(nested_video_embedding, dict):
                segments = nested_video_embedding.get("segments")
            if isinstance(segments, list):
                return segments

    return []


def _vector_mean(vectors: list[list[float]]) -> list[float] | None:
    if not vectors:
        return None
    dimension = len(vectors[0])
    if dimension == 0:
        return None
    valid_vectors = [vector for vector in vectors if len(vector) == dimension]
    if not valid_vectors:
        return None
    totals = [0.0] * dimension
    for vector in valid_vectors:
        for index, value in enumerate(vector):
            totals[index] += value
    count = float(len(valid_vectors))
    return [value / count for value in totals]


def _normalize_vector(vector: list[float]) -> list[float] | None:
    squared_sum = sum(value * value for value in vector)
    if squared_sum <= 0.0:
        return None
    norm = math.sqrt(squared_sum)
    return [value / norm for value in vector]


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(left * right for left, right in zip(a, b))


def _parse_reel_doc(raw_doc: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_doc)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def compute_indexed_asset_embedding_signature(
    client: Any,
    index_id: str,
    indexed_asset_id: str,
) -> dict[str, Any] | None:
    retrieve_fn = client.indexes.indexed_assets.retrieve
    accepted_kwargs: set[str] = set()
    has_var_keyword = False
    try:
        signature = inspect.signature(retrieve_fn)
        accepted_kwargs = set(signature.parameters.keys())
        has_var_keyword = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        # Some SDK/client wrappers may not expose signatures.
        accepted_kwargs = set()
        has_var_keyword = True

    supports_embedding_option = (
        "embedding_option" in accepted_kwargs
        or has_var_keyword
    )
    supports_embed = (
        "embed" in accepted_kwargs
        or has_var_keyword
    )

    attempts: list[tuple[str, dict[str, Any]]] = []
    if supports_embedding_option:
        # "visual-text" is not supported on many indexes; probe only commonly allowed modes.
        for option in ("visual", "audio", "transcription"):
            # Keep both payload shapes for SDK/API compatibility.
            attempts.append((f"{option}-list", {"embedding_option": [option]}))
            attempts.append((f"{option}-str", {"embedding_option": option}))
    if supports_embed:
        attempts.append(("legacy-embed", {"embed": True}))
    attempts.append(("default", {}))

    last_error: Exception | None = None

    for embedding_option_name, kwargs in attempts:
        try:
            details = retrieve_fn(
                index_id=index_id,
                indexed_asset_id=indexed_asset_id,
                **kwargs,
            )
        except TypeError as exc:
            # Avoid noisy failures when running across mixed SDK versions.
            if "unexpected keyword argument" in str(exc):
                continue
            last_error = exc
            continue
        except Exception as exc:
            error_text = str(exc).lower()
            # Unsupported embedding options should not fail the whole job.
            if "parameter_invalid" in error_text and "embedding_option" in error_text:
                continue
            last_error = exc
            continue

        segments = _extract_video_embedding_segments(details)
        vectors: list[list[float]] = []
        for segment in segments:
            vector = _extract_segment_embedding_vector(segment)
            if vector:
                vectors.append(vector)
        pooled = _vector_mean(vectors)
        if pooled is None:
            continue
        normalized = _normalize_vector(pooled)
        if normalized is None:
            continue
        return {
            "embedding_option": embedding_option_name,
            "dimension": len(normalized),
            "segment_count": len(vectors),
            "vector": normalized,
        }

    if last_error:
        print(f"Embedding signature retrieval failed for {indexed_asset_id}: {last_error}")
    return None


def _extract_analysis_vector_from_reel_doc(doc: dict[str, Any]) -> list[float] | None:
    analysis = doc.get("analysis")
    if not isinstance(analysis, dict):
        return None
    embedding_signature = analysis.get("embedding_signature")
    if not isinstance(embedding_signature, dict):
        return None
    return _coerce_float_vector(embedding_signature.get("vector"))


def find_similar_reel_in_datastore(
    datastore_client: valkey.Valkey,
    *,
    key_prefix: str,
    current_reel_id: str,
    current_index_id: str | None,
    current_vector: list[float],
    similarity_threshold: float,
    candidate_limit: int,
) -> dict[str, Any] | None:
    pattern = f"{key_prefix}*" if key_prefix else "*"
    current_key = f"{key_prefix}{current_reel_id}"
    best_match: dict[str, Any] | None = None
    inspected = 0

    for key in datastore_client.scan_iter(match=pattern, count=200):
        if inspected >= candidate_limit:
            break
        if key == current_key:
            continue
        try:
            raw_doc = datastore_client.get(key)
        except valkey.exceptions.ResponseError:
            continue
        if not raw_doc:
            continue
        doc = _parse_reel_doc(raw_doc)
        if doc is None:
            continue
        doc = normalize_reel_document(reel_id=doc.get("reel_id") or key.replace(key_prefix, "", 1), raw_doc=doc)

        candidate_analysis = doc.get("analysis")
        if not isinstance(candidate_analysis, dict):
            continue
        if not candidate_analysis.get("twelvelabs"):
            continue
        candidate_indexing = candidate_analysis.get("indexing")
        candidate_index_id = None
        if isinstance(candidate_indexing, dict):
            candidate_index_id = candidate_indexing.get("index_id")
        if current_index_id and candidate_index_id and current_index_id != candidate_index_id:
            continue

        candidate_vector = _extract_analysis_vector_from_reel_doc(doc)
        if not candidate_vector:
            continue
        if len(candidate_vector) != len(current_vector):
            continue

        inspected += 1
        similarity = _dot_product(current_vector, candidate_vector)
        if best_match is None or similarity > best_match["similarity_score"]:
            best_match = {
                "reel_id": doc.get("reel_id"),
                "analysis": candidate_analysis.get("twelvelabs"),
                "indexing": candidate_analysis.get("indexing"),
                "similarity_score": similarity,
                "source_key": key,
            }

    if not best_match:
        return None
    if best_match["similarity_score"] < similarity_threshold:
        return None
    return best_match


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
    embedding_signature = result.get("embedding_signature")
    similarity_match = result.get("similarity_match")
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
    current_dict["analysis"]["embedding_signature"] = embedding_signature
    current_dict["analysis"]["similarity_match"] = similarity_match
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
    embedding_signature = compute_indexed_asset_embedding_signature(
        client=client,
        index_id=indexing_result.index_id,
        indexed_asset_id=indexing_result.indexed_asset_id,
    )

    similar_match = None
    if (
        config.similarity_reuse_enabled
        and embedding_signature
        and isinstance(embedding_signature.get("vector"), list)
        and len(embedding_signature["vector"]) > 0
    ):
        similar_match = find_similar_reel_in_datastore(
            datastore_client=datastore_client,
            key_prefix=config.datastore_key_prefix,
            current_reel_id=reel_id,
            current_index_id=indexing_result.index_id,
            current_vector=embedding_signature["vector"],
            similarity_threshold=config.similarity_reuse_threshold,
            candidate_limit=config.similarity_candidate_limit,
        )

    if similar_match:
        analysis_result = similar_match["analysis"]
        similarity_match = {
            "reused": True,
            "matched_reel_id": similar_match["reel_id"],
            "similarity_score": similar_match["similarity_score"],
            "threshold": config.similarity_reuse_threshold,
            "strategy": "marengo_embedding_cosine",
            "matched_source_key": similar_match["source_key"],
        }
        print(
            "Reused existing analysis from "
            f"reel_id={similar_match['reel_id']} similarity={similar_match['similarity_score']:.4f}"
        )
    else:
        analysis_result = analyze_video_for_ai(
            client=client,
            video_id=indexing_result.indexed_asset_id,
            prompt=config.analysis_prompt,
        )
        similarity_match = {
            "reused": False,
            "threshold": config.similarity_reuse_threshold,
            "strategy": "marengo_embedding_cosine",
        }

    result = {
        "job_id": job_id,
        "reel_url": reel_url,
        "reel_id": reel_id,
        "downloaded_file": str(downloaded_path),
        "indexing": asdict(indexing_result),
        "analysis": analysis_result,
        "embedding_signature": embedding_signature,
        "similarity_match": similarity_match,
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
