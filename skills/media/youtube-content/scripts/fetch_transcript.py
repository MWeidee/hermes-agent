#!/usr/bin/env python3
"""
Fetch a YouTube video transcript and output it as structured JSON.

Usage:
    python fetch_transcript.py <url_or_video_id> [--language en,tr] [--timestamps]

Output (JSON):
    {
        "video_id": "...",
        "language": "en",
        "segments": [{"text": "...", "start": 0.0, "duration": 2.5}, ...],
        "full_text": "complete transcript as plain text",
        "timestamped_text": "00:00 first line\n00:05 second line\n..."
    }

Install dependency:  pip install youtube-transcript-api
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character video ID from various YouTube URL formats."""
    url_or_id = url_or_id.strip()
    patterns = [
        r'(?:v=|youtu\.be/|shorts/|embed/|live/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fetch_transcript(video_id: str, languages: list = None):
    """Fetch transcript segments from YouTube.

    Returns a list of dicts with 'text', 'start', and 'duration' keys.
    Compatible with youtube-transcript-api v1.x.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("Error: youtube-transcript-api not installed. Run: pip install youtube-transcript-api",
              file=sys.stderr)
        sys.exit(1)

    api = YouTubeTranscriptApi()
    if languages:
        result = api.fetch(video_id, languages=languages)
    else:
        result = api.fetch(video_id)

    # v1.x returns FetchedTranscriptSnippet objects; normalize to dicts
    return [
        {"text": seg.text, "start": seg.start, "duration": seg.duration}
        for seg in result
    ]


def _repo_root() -> Path:
    """Return the Hermes repo root for importing shared tool modules from this skill script."""
    return Path(__file__).resolve().parents[4]


def _download_youtube_audio(url_or_id: str, output_dir: str) -> str:
    """Download a YouTube video's best small audio stream into ``output_dir``.

    This intentionally avoids yt-dlp post-processing, so the fallback does not
    require ffmpeg merely to obtain an audio file. The STT backend can convert
    later if its selected provider needs a WAV input.
    """
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        raise RuntimeError("Audio fallback requires yt-dlp to download YouTube audio")

    output_dir_path = Path(output_dir)
    before = {p.resolve() for p in output_dir_path.glob("*") if p.is_file()}
    command = [
        yt_dlp,
        "--no-playlist",
        "--max-filesize",
        "25M",
        "-f",
        "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio",
        "-o",
        str(output_dir_path / "%(id)s.%(ext)s"),
        url_or_id,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"yt-dlp audio download failed: {details}") from exc

    downloaded = [
        p for p in output_dir_path.glob("*")
        if p.is_file() and p.resolve() not in before
    ]
    audio_files = [
        p for p in downloaded
        if p.suffix.lower() in {".mp3", ".mp4", ".m4a", ".webm", ".ogg", ".aac", ".flac", ".wav"}
    ]
    if not audio_files:
        raise RuntimeError("yt-dlp completed but no audio file was downloaded")
    audio_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(audio_files[0])


def fetch_audio_fallback_transcript(url_or_id: str, model: Optional[str] = None) -> dict:
    """Download YouTube audio and transcribe it with Hermes STT/Whisper providers."""
    repo = str(_repo_root())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    from tools.transcription_tools import transcribe_audio

    with tempfile.TemporaryDirectory(prefix="hermes-youtube-audio-") as output_dir:
        audio_path = _download_youtube_audio(url_or_id, output_dir)
        result = transcribe_audio(audio_path, model=model)

    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Audio transcription failed")

    transcript = (result.get("transcript") or "").strip()
    if not transcript:
        raise RuntimeError("Audio transcription returned an empty transcript")

    return {
        "video_id": extract_video_id(url_or_id),
        "language": "audio",
        "source": "audio_fallback",
        "provider": result.get("provider", "unknown"),
        "segment_count": 1,
        "duration": "0:00",
        "segments": [{"text": transcript, "start": 0.0, "duration": 0.0}],
        "full_text": transcript,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube transcript as JSON")
    parser.add_argument("url", help="YouTube URL or video ID")
    parser.add_argument("--language", "-l", default=None,
                        help="Comma-separated language codes (e.g. en,tr). Default: auto")
    parser.add_argument("--timestamps", "-t", action="store_true",
                        help="Include timestamped text in output")
    parser.add_argument("--text-only", action="store_true",
                        help="Output plain text instead of JSON")
    parser.add_argument("--audio-fallback", action="store_true",
                        help="If caption transcript fetching fails, download audio with yt-dlp and transcribe via Hermes STT/Whisper")
    parser.add_argument("--stt-model", default=None,
                        help="Optional STT model override for --audio-fallback")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    languages = [l.strip() for l in args.language.split(",")] if args.language else None

    try:
        segments = fetch_transcript(video_id, languages)
    except Exception as e:
        if args.audio_fallback:
            try:
                fallback_result = fetch_audio_fallback_transcript(args.url, model=args.stt_model)
                if args.text_only:
                    print(fallback_result["full_text"])
                else:
                    print(json.dumps(fallback_result, ensure_ascii=False, indent=2))
                return
            except Exception as fallback_error:
                print(json.dumps({
                    "error": str(e),
                    "audio_fallback_error": str(fallback_error),
                }, ensure_ascii=False))
                sys.exit(1)

        error_msg = str(e)
        if "disabled" in error_msg.lower():
            print(json.dumps({"error": "Transcripts are disabled for this video."}))
        elif "no transcript" in error_msg.lower():
            print(json.dumps({"error": f"No transcript found. Try specifying a language with --language."}))
        else:
            print(json.dumps({"error": error_msg}))
        sys.exit(1)

    full_text = " ".join(seg["text"] for seg in segments)
    timestamped = "\n".join(
        f"{format_timestamp(seg['start'])} {seg['text']}" for seg in segments
    )

    if args.text_only:
        print(timestamped if args.timestamps else full_text)
        return

    result = {
        "video_id": video_id,
        "segment_count": len(segments),
        "duration": format_timestamp(segments[-1]["start"] + segments[-1]["duration"]) if segments else "0:00",
        "full_text": full_text,
    }
    if args.timestamps:
        result["timestamped_text"] = timestamped

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
