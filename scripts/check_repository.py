"""Reject ignored or oversized files in the Git index before committing.

This is a file policy check, not a content-based secret scanner.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main() -> int:
    tracked = [os.fsdecode(p) for p in git("ls-files", "--cached", "-z").split(b"\0") if p]
    ignored = [os.fsdecode(p) for p in git("ls-files", "--cached", "--ignored", "--exclude-standard", "-z").split(b"\0") if p]
    problems = [f"Ignored file is staged/tracked: {path}" for path in ignored]
    for path in tracked:
        size = int(git("cat-file", "-s", f":{path}"))
        if size > MAX_BYTES:
            problems.append(f"File exceeds 10 MiB: {path} ({size} bytes)")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Git file policy passed: {len(tracked)} indexed files, none ignored or over 10 MiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
