"""Reject ignored or oversized files in the Git index before committing.

This is a file policy check, not a content-based secret scanner.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also inspect all reachable Git objects")
    args = parser.parse_args()
    tracked = [os.fsdecode(p) for p in git("ls-files", "--cached", "-z").split(b"\0") if p]
    ignored = [os.fsdecode(p) for p in git("ls-files", "--cached", "--ignored", "--exclude-standard", "-z").split(b"\0") if p]
    problems = [f"Ignored file is staged/tracked: {path}" for path in ignored]
    for path in tracked:
        size = int(git("cat-file", "-s", f":{path}"))
        if size > MAX_BYTES:
            problems.append(f"File exceeds 10 MiB: {path} ({size} bytes)")
    if args.history:
        objects = git("rev-list", "--objects", "--all")
        checked = subprocess.check_output(
            ["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
            input=objects, cwd=ROOT,
        )
        blobs = 0
        for line in checked.decode("utf-8", errors="replace").splitlines():
            kind, size, *names = line.split(" ", 2)
            if kind != "blob":
                continue
            blobs += 1
            name = names[0] if names else "<unnamed>"
            if int(size) > MAX_BYTES:
                problems.append(f"Historical blob exceeds 10 MiB: {name} ({size} bytes)")
            if name.startswith(("data/raw/", "data/manifests/")) and not name.endswith((".dvc", "/.gitignore")):
                problems.append(f"Raw data/manifest payload in Git history: {name}")
        print(f"Inspected {blobs} reachable historical Git blobs (all local refs).")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Git file policy passed: {len(tracked)} indexed files, none ignored or over 10 MiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
