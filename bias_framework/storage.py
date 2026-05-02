from __future__ import annotations

import csv
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from bias_framework.clients.base import ScoreResult

CSV_FIELDS = [
    "model_id", "provider", "condition_id", "trial_number",
    "score", "score_technical", "score_business", "score_experience",
    "score_communication", "score_education",
    "reasoning", "raw_response", "latency_ms", "timestamp", "error",
]

DIMENSION_COLS = [
    "score_technical", "score_business", "score_experience",
    "score_communication", "score_education",
]

_write_lock = threading.Lock()
_header_written: set[Path] = set()


def _try_int(val: str | None) -> Optional[int]:
    try:
        return int(val) if val else None
    except (ValueError, TypeError):
        return None


def append_result(result: ScoreResult, path: Path) -> None:
    with _write_lock:
        write_header = path not in _header_written and (not path.exists() or path.stat().st_size == 0)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                _header_written.add(path)
            row = asdict(result)
            if row["raw_response"] and len(row["raw_response"]) > 2000:
                row["raw_response"] = row["raw_response"][:2000] + "…"
            writer.writerow(row)


def load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return set of (model_id, condition_id, trial_number) already in path."""
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("score"):  # only count rows that succeeded
                    keys.add((row["model_id"], row["condition_id"], int(row["trial_number"])))
            except (KeyError, ValueError):
                continue
    return keys


def load_results(path: Path) -> list[dict]:
    """Load all rows from a CSV, casting numeric fields."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["score"]        = _try_int(row.get("score"))
            row["trial_number"] = _try_int(row.get("trial_number"))
            row["latency_ms"]   = _try_int(row.get("latency_ms"))
            for col in DIMENSION_COLS:
                row[col] = _try_int(row.get(col))
            rows.append(row)
    return rows
