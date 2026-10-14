"""Download the pinned public checkpoint without Git or mutable revisions."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from signai.ptbxl import sha256, write_json


def fetch_bytes(url, limit):
    with urlopen(Request(url, headers={"User-Agent": "signai-probe/0.1"}), timeout=60) as response:
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("Source metadata exceeded size limit")
    return payload


def main():
    root = Path.cwd()
    config = json.loads((root / "docs/moment_probe_config.json").read_text(encoding="utf-8"))
    model, revision = config["model_id"], config["checkpoint_revision"]
    source = root / "artifacts/moment-probe/source"
    destination = root / "artifacts/moment-probe/checkpoint"
    source.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(fetch_bytes(f"https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true", 1048576))
    if manifest["sha"] != revision:
        raise ValueError("Checkpoint revision mismatch")
    files = {item["rfilename"]: item for item in manifest["siblings"]}
    payload = fetch_bytes(f"https://huggingface.co/{model}/resolve/{revision}/config.json", 65536)
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if blob != files["config.json"]["blobId"]:
        raise ValueError("Configuration blob does not match pinned revision")
    (destination / "config.json").write_bytes(payload)
    (source / "checkpoint_config.json").write_bytes(payload)
    weight_file = destination / "model.safetensors"
    expected = files["model.safetensors"]["lfs"]["sha256"]
    if not weight_file.exists() or sha256(weight_file) != expected:
        temporary = weight_file.with_suffix(".part")
        try:
            url = f"https://huggingface.co/{model}/resolve/{revision}/model.safetensors"
            with urlopen(Request(url, headers={"User-Agent": "signai-probe/0.1"}), timeout=60) as response, temporary.open("wb") as stream:
                count = 0
                while chunk := response.read(1048576):
                    count += len(chunk)
                    if count > files["model.safetensors"]["size"]:
                        raise ValueError("Checkpoint exceeds pinned byte size")
                    stream.write(chunk)
            if sha256(temporary) != expected:
                raise ValueError("Checkpoint SHA256 mismatch")
            temporary.replace(weight_file)
        finally:
            temporary.unlink(missing_ok=True)
    write_json(source / "checkpoint_manifest.json", manifest)
    print(f"Verified {model}@{revision}: {weight_file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
