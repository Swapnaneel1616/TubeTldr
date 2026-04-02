"""Fetch captions via youtube-transcript-api (same pattern as test.ipynb)."""

from __future__ import annotations

from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi


def fetch_transcript_text(video_id: str, languages: tuple[str, ...] | None = None) -> str:
    """
    Returns full transcript as a single string.
    Raises TranscriptsDisabled if captions are off, or other exceptions from the API.
    """
    from . import config

    langs = list(languages or config.TRANSCRIPT_LANGUAGES)
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id, languages=langs)
    return " ".join(snippet.text for snippet in fetched.snippets)


def load_transcript_safe(video_id: str, languages: tuple[str, ...] | None = None):
    """Returns (text, None) on success or (None, error_message) on failure."""
    try:
        text = fetch_transcript_text(video_id, languages=languages)
        if not text.strip():
            return None, "Transcript was empty."
        return text, None
    except TranscriptsDisabled:
        return None, "No captions available for this video."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"
