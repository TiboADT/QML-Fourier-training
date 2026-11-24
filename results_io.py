"""Shared CSV-append infrastructure for frame_potential.py and
tracking.py: both append one row per run to a CSV, creating it
with a header if missing, and both need the commit that produced a row
traceable later. Previously duplicated -- frame_potential.save_estimate
had its own inline writer, tracking.py its own
_ensure_csv/_append_row, doing the exact same thing -- unified here, with
a header-mismatch check added (code review, "must fix": local CSVs were
found column-shifted after fields were added to a field list but not to
already-written headers, with nothing catching it).
"""

import csv
import os
import subprocess
from pathlib import Path


def git_commit_hash() -> str:
    """So a saved row can always be traced back to the code that produced
    it -- "unknown" outside a git repo or with no commits."""
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
    than `fieldnames` -- silently writing a row under the wrong columns
    (what plain appending does) is exactly how the column-shifted CSVs
    this review found happened: fields were added to a field list without
    updating already-written files, and nothing caught it."""
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
