"""Cached HTTPS download helper shared by load_graph.py and map_sensors.py.

Real remote files only — see data/README.md for provenance of every URL this
project fetches. Never falls back to generating placeholder content: a
failed download raises, so `make data` fails loudly instead of silently
proceeding on missing data (CLAUDE.md rule 1).
"""

from __future__ import annotations

from pathlib import Path

import requests


def fetch(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download `url` to `dest` unless `dest` already exists.

    Caching is by filename only (no ETag/checksum tracking) since every URL
    here points at a fixed, versioned dataset file, not a moving target. Uses
    `requests` (rather than bare `urllib`) so TLS verification goes through
    certifi's CA bundle, which is what's reliably present in this project's
    venv across platforms.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.rename(dest)
    return dest
