"""Tests for skills/media/youtube-content/scripts/fetch_transcript.py (issue #22243)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "media" / "youtube-content" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fetch_transcript


class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert fetch_transcript.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_video_id(self):
        assert fetch_transcript.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_with_extra_params(self):
        assert fetch_transcript.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42") == "dQw4w9WgXcQ"


class TestFormatTimestamp:
    def test_seconds_only(self):
        assert fetch_transcript.format_timestamp(90) == "1:30"

    def test_with_hours(self):
        assert fetch_transcript.format_timestamp(3661) == "1:01:01"

    def test_zero(self):
        assert fetch_transcript.format_timestamp(0) == "0:00"

    def test_minutes_only(self):
        assert fetch_transcript.format_timestamp(600) == "10:00"


class TestFetchTranscriptImportError:
    def test_missing_dep_exits_with_message(self, capsys):
        """fetch_transcript exits with code 1 and prints install hint when package missing (issue #22243)."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("No module named 'youtube_transcript_api'")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(SystemExit) as exc_info:
                fetch_transcript.fetch_transcript("dQw4w9WgXcQ")
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "youtube-transcript-api" in captured.err


class TestAudioFallback:
    def test_audio_fallback_downloads_audio_and_transcribes(self, tmp_path, monkeypatch):
        """When captions are blocked, the helper can download audio and use Hermes STT."""
        audio_path = tmp_path / "video.m4a"

        def fake_run(cmd, check, capture_output, text):
            assert cmd[0] == "yt-dlp"
            assert "--no-playlist" in cmd
            assert "--max-filesize" in cmd
            audio_path.write_bytes(b"fake m4a audio")
            return mock.Mock(stdout="", stderr="")

        monkeypatch.setattr(fetch_transcript.shutil, "which", lambda name: "yt-dlp" if name == "yt-dlp" else None)
        monkeypatch.setattr(fetch_transcript.subprocess, "run", fake_run)
        monkeypatch.setattr(fetch_transcript.tempfile, "TemporaryDirectory", lambda prefix=None: _StaticTempDir(tmp_path))
        monkeypatch.setitem(
            sys.modules,
            "tools.transcription_tools",
            mock.Mock(transcribe_audio=mock.Mock(return_value={
                "success": True,
                "transcript": "audio transcript",
                "provider": "local_command",
            })),
        )

        result = fetch_transcript.fetch_audio_fallback_transcript("https://youtu.be/dQw4w9WgXcQ")

        assert result["source"] == "audio_fallback"
        assert result["provider"] == "local_command"
        assert result["full_text"] == "audio transcript"
        assert result["segments"] == [{"text": "audio transcript", "start": 0.0, "duration": 0.0}]

    def test_audio_fallback_reports_missing_ytdlp(self, monkeypatch):
        monkeypatch.setattr(fetch_transcript.shutil, "which", lambda name: None)

        with pytest.raises(RuntimeError, match="yt-dlp"):
            fetch_transcript.fetch_audio_fallback_transcript("dQw4w9WgXcQ")


class _StaticTempDir:
    def __init__(self, path):
        self.path = str(path)

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc, tb):
        return False


class TestPyprojectDeclaresYoutubeExtra:
    def test_youtube_extra_declared_in_pyproject(self):
        """youtube-transcript-api must be listed in pyproject.toml [youtube] extra (issue #22243)."""
        import tomllib
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        extras = data.get("project", {}).get("optional-dependencies", {})
        assert "youtube" in extras, "Missing [youtube] extra in pyproject.toml"
        youtube_deps = " ".join(extras["youtube"])
        assert "youtube-transcript-api" in youtube_deps

    def test_youtube_extra_included_in_all(self):
        """[all] extra must include hermes-agent[youtube] (issue #22243)."""
        import tomllib
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject_path.open("rb") as f:
            data = tomllib.load(f)
        all_deps = " ".join(data["project"]["optional-dependencies"].get("all", []))
        assert "youtube" in all_deps, "[all] extra does not include hermes-agent[youtube]"
