"""Resolve an 11-character YouTube video id from a URL or raw id."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


def extract_video_id(url_or_id: str) -> str:
    raw = (url_or_id or "").strip()
    if not raw:
        raise ValueError("Please enter a YouTube URL or video ID.")

    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", raw):
        return raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        seg = parsed.path.strip("/").split("/")[0]
        if seg and re.fullmatch(r"[a-zA-Z0-9_-]{11}", seg):
            return seg

    if host.endswith("youtube.com"):
        q = parse_qs(parsed.query)
        if "v" in q and q["v"]:
            vid = q["v"][0]
            if re.fullmatch(r"[a-zA-Z0-9_-]{11}", vid):
                return vid
        parts = [p for p in parsed.path.split("/") if p]
        for key in ("embed", "shorts", "live"):
            if key in parts:
                i = parts.index(key)
                if i + 1 < len(parts):
                    cand = parts[i + 1]
                    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cand):
                        return cand

    raise ValueError("Could not find a valid 11-character YouTube video ID in that input.")
