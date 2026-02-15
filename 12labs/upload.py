#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from twelvelabs import TwelveLabs


@dataclass(frozen=True)
class IndexingResult:
    asset_id: str
    indexed_asset_id: str
    index_id: str
    status: str
    stream_status: str | None = None


def load_environment(env_file: str | Path | None = None) -> None:
    if env_file:
        load_dotenv(dotenv_path=Path(env_file), override=False)
        return

    script_dir_env = Path(__file__).resolve().with_name(".env")
    if script_dir_env.exists():
        load_dotenv(dotenv_path=script_dir_env, override=False)
    load_dotenv(override=False)


def resolve_api_key(
    api_key: str | None = None,
    env_var: str = "TWELVE_LABS_API_KEY",
    env_file: str | Path | None = None,
) -> str:
    load_environment(env_file)

    if api_key:
        return api_key

    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"Missing API key. Set {env_var} or provide --api-key.")
    return key


def resolve_index_id(
    index_id: str | None = None,
    env_var: str = "TWELVE_LABS_INDEX_ID",
    env_file: str | Path | None = None,
) -> str:
    load_environment(env_file)

    if index_id:
        return index_id

    resolved_index_id = os.getenv(env_var)
    if not resolved_index_id:
        raise RuntimeError(f"Missing index ID. Set {env_var} or provide --index-id.")
    return resolved_index_id


def create_client(
    api_key: str | None = None,
    env_var: str = "TWELVE_LABS_API_KEY",
    env_file: str | Path | None = None,
) -> TwelveLabs:
    key = resolve_api_key(api_key=api_key, env_var=env_var, env_file=env_file)
    return TwelveLabs(api_key=key)


def upload_and_index_video(
    video_path: str | Path,
    index_id: str | None = None,
    *,
    api_key: str | None = None,
    env_file: str | Path | None = None,
    poll_interval: int = 5,
    wait_for_stream: bool = True,
) -> IndexingResult:
    client = create_client(api_key=api_key, env_file=env_file)
    resolved_index_id = resolve_index_id(index_id=index_id, env_file=env_file)

    resolved_video_path = Path(video_path).expanduser().resolve()
    if not resolved_video_path.exists() or not resolved_video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {resolved_video_path}")

    with resolved_video_path.open("rb") as file_handle:
        asset = client.assets.create(
            method="direct",
            file=file_handle,
            filename=resolved_video_path.name,
        )

    asset_id = getattr(asset, "id", None)
    if not asset_id:
        raise RuntimeError("Asset upload did not return an asset ID")

    while True:
        asset = client.assets.retrieve(asset_id)
        status = (getattr(asset, "status", None) or "").lower()
        if status == "ready":
            break
        if status == "failed":
            raise RuntimeError(f"Asset processing failed. asset_id={asset_id}, status={asset.status}")
        time.sleep(poll_interval)

    indexed_asset = client.indexes.indexed_assets.create(
        index_id=resolved_index_id,
        asset_id=asset_id,
        enable_video_stream=True,
    )

    indexed_asset_id = getattr(indexed_asset, "id", None)
    if not indexed_asset_id:
        raise RuntimeError("Indexing did not return an indexed asset ID")

    while True:
        details = client.indexes.indexed_assets.retrieve(
            index_id=resolved_index_id,
            indexed_asset_id=indexed_asset_id,
        )
        status = (getattr(details, "status", None) or "").lower()

        if status == "ready":
            if wait_for_stream:
                hls = getattr(details, "hls", None)
                hls_status = (getattr(hls, "status", None) or "").upper()
                if hls_status and hls_status != "COMPLETE":
                    time.sleep(poll_interval)
                    continue

            return IndexingResult(
                asset_id=asset_id,
                indexed_asset_id=indexed_asset_id,
                index_id=resolved_index_id,
                status=details.status,
                stream_status=getattr(getattr(details, "hls", None), "status", None),
            )

        if status == "failed":
            raise RuntimeError(
                f"Indexed asset failed. indexed_asset_id={indexed_asset_id}, status={details.status}"
            )

        time.sleep(poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple Two-Step TwelveLabs upload: 1) create asset, 2) index asset"
    )
    parser.add_argument("video_path", help="Path to local video file")
    parser.add_argument("--index-id", default=None, help="TwelveLabs index ID (optional)")
    parser.add_argument("--api-key", default=None, help="TwelveLabs API key (optional)")
    parser.add_argument("--env-file", default=None, help="Path to .env file (optional)")
    parser.add_argument("--poll-interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument(
        "--no-wait-for-stream",
        action="store_true",
        help="Return as soon as indexing is ready without waiting for stream encoding",
    )

    args = parser.parse_args()

    try:
        result = upload_and_index_video(
            video_path=args.video_path,
            index_id=args.index_id,
            api_key=args.api_key,
            env_file=args.env_file,
            poll_interval=args.poll_interval,
            wait_for_stream=not args.no_wait_for_stream,
        )
        print("Indexing complete")
        print(f"index_id={result.index_id}")
        print(f"asset_id={result.asset_id}")
        print(f"indexed_asset_id={result.indexed_asset_id}")
        print(f"status={result.status}")
        print(f"stream_status={result.stream_status}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
