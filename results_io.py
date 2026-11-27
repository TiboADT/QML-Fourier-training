"""Shared CSV-append infrastructure for frame_potential.py and
tracking.py: both append one row per run to a CSV, creating it
with a header if missing, and both need the commit that produced a row
traceable later.
"""

import csv
import os
import subprocess
from pathlib import Path


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def ensure_csv(path: str, fieldnames: list) -> None:
    """Create `path` with a header row if it does not exist yet -- for
    load-side callers that need the file to exist before reading it."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_row(path: str, fieldnames: list, row: dict) -> None:
    """Append one row to `path`, creating it with a header if it doesn't
    exist yet. Raises if the file already exists with a DIFFERENT header
    than `fieldnames`"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != fieldnames:
            raise ValueError(
                f"{path}'s existing header does not match the current field "
                f"list -- refusing to append under the wrong columns.\n"
                f"  existing: {existing_header}\n"
                f"  current:  {fieldnames}"
            )
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
