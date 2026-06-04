"""Reproducibility metadata: git SHA, package versions, map hashes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git",) + args, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL, text=True
        )
        return sorted(line for line in out.splitlines() if line.strip())
    except Exception:  # noqa: BLE001
        return []


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _map_hashes(map_dir: str = "./maps") -> dict[str, str]:
    p = Path(map_dir)
    if not p.exists():
        return {}
    return {f.name: _hash_file(f) for f in sorted(p.glob("*.pickle"))}


def snapshot(out_dir: str, extra: dict | None = None) -> dict:
    """Write provenance.json into ``out_dir`` and return the dict."""
    os.makedirs(out_dir, exist_ok=True)
    data = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pip_freeze": _pip_freeze(),
        "map_hashes": _map_hashes(),
        "extra": extra or {},
    }
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data
