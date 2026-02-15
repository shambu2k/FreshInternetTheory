#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse


def extract_reel_id(url: str) -> str:
	parsed = urlparse(url)
	path = parsed.path.rstrip("/")

	match = re.search(r"/reels?/(\d+)", path)
	if match:
		return match.group(1)

	raise ValueError(
		"Could not extract reel_id from URL. Expected format like: "
		"https://www.facebook.com/reel/1031867656387604"
	)


def download_reel(url: str, reel_id: str, output_dir: str | Path | None = None) -> Path:
	if shutil.which("yt-dlp") is None:
		raise RuntimeError("yt-dlp is not found in PATH")

	if output_dir is None:
		output_dir_path = Path(__file__).resolve().parent / "downloads"
	else:
		output_dir_path = Path(output_dir).expanduser().resolve()

	output_dir_path.mkdir(parents=True, exist_ok=True)
	output_template = str(output_dir_path / f"{reel_id}.%(ext)s")

	cmd = [
		"yt-dlp",
		"--no-playlist",
		"--format",
		"b[height<=480]/b/bv*[height<=480]+ba/bv*+ba",
		"--merge-output-format",
		"mp4",
		"--remux-video",
		"mp4",
		"--output",
		output_template,
		url,
	]

	result = subprocess.run(cmd)
	if result.returncode != 0:
		raise RuntimeError("Download failed. If merging fails, ensure ffmpeg is installed.")

	return output_dir_path / f"{reel_id}.mp4"


def download_facebook_reel(url: str, output_dir: str | Path | None = None) -> Path:
	reel_id = extract_reel_id(url)
	return download_reel(url, reel_id, output_dir=output_dir)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Download a Facebook Reel as [reel_id].mp4 using yt-dlp"
	)
	parser.add_argument("url", help="Facebook reel URL")
	parser.add_argument(
		"-o",
		"--output-dir",
		help="Directory to save the downloaded reel (default: 12labs/downloads)",
		default=None,
	)
	args = parser.parse_args()

	try:
		file_path = download_facebook_reel(args.url, output_dir=args.output_dir)
		print(f"Downloaded: {file_path}")
		return 0
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
