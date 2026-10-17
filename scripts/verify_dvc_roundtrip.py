"""Pull into an independent clone, first proving that an empty remote fails."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

from ptbxl_snapshot import MANIFEST, RAW, verify

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["data/raw/ptb-xl/1.0.3.dvc", "data/manifests/ptbxl-1.0.3.json.dvc"]


def run(args, cwd, check=True):
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError(f"{args}:\n{result.stdout}\n{result.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", type=Path, required=True, help="Existing local DVC object store")
    parser.add_argument("--working-tree-snapshot", action="store_true", help="Overlay pending metadata; NOT a committed clone test")
    args = parser.parse_args()
    remote = args.remote.resolve()
    if not remote.is_dir():
        parser.error("--remote must be an existing directory")
    source = verify(ROOT)
    lab = ROOT / ".dvc-lab" / uuid4().hex
    lab.mkdir(parents=True)
    clone = lab / "clone"
    run(["git", "clone", "--no-local", str(ROOT), str(clone)], ROOT)
    if args.working_tree_snapshot:
        for name in [".dvc/config", ".dvc/.gitignore", ".dvcignore", ".gitignore",
                     "data/raw/ptb-xl/.gitignore", "docs/provenance.json", *TARGETS]:
            target = clone / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    dvc = [sys.executable, "-m", "dvc"]
    # Explicit private cache: the clone must not reuse this project's warm cache.
    run([*dvc, "config", "--local", "cache.dir", str(clone / ".dvc/cache")], clone)
    if (clone / RAW).exists() or (clone / MANIFEST).exists() or (clone / ".dvc/cache").exists():
        raise RuntimeError("Clone unexpectedly already contains artifacts or cache")
    empty = lab / "empty-remote"
    empty.mkdir()
    run([*dvc, "remote", "add", "--local", "-d", "roundtrip", str(empty)], clone)
    broken = run([*dvc, "pull", *TARGETS], clone, check=False)
    (lab / "broken-remote.log").write_text(broken.stdout + broken.stderr, encoding="utf-8")
    if broken.returncode == 0 or "missing" not in (broken.stdout + broken.stderr).lower():
        raise RuntimeError("Expected missing-artifact failure; inspect broken-remote.log")
    run([*dvc, "remote", "modify", "--local", "roundtrip", "url", str(remote)], clone)
    pulled = run([*dvc, "pull", *TARGETS], clone)
    (lab / "pull.log").write_text(pulled.stdout + pulled.stderr, encoding="utf-8")
    restored = verify(clone)
    if source != restored:
        raise RuntimeError("Restored checksums differ")
    report = {"mode": "HEAD clone plus pending metadata overlay" if args.working_tree_snapshot else "committed HEAD clone",
              "git_head": run(["git", "rev-parse", "HEAD"], clone).stdout.strip(),
              "clone": str(clone), "remote": str(remote), "cache_initially_empty": True,
              "broken_remote_exit_code": broken.returncode, "restored": restored,
              "source_equals_clone": True}
    (lab / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
