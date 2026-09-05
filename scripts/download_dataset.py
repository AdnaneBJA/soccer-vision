"""Download the public CC BY 4.0 soccer dataset mirror with revision/hash provenance."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests
import yaml

REPO = "martinjolif/football-player-detection"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/soccer"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    api = f"https://huggingface.co/api/datasets/{REPO}"
    response = requests.get(api + "/revision/" + args.revision, timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    url = api + f"/tree/{revision}?recursive=true&limit=1000"
    entries = []
    while url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        entries.extend(entry for entry in response.json() if entry["type"] == "file" and
                       (entry["path"].startswith("data/") or entry["path"] == "README.md"))
        url = response.links.get("next", {}).get("url")
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(entry: dict) -> dict:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe dataset path")
        local = args.output / relative
        remote = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{relative.as_posix()}"
        expected = entry.get("lfs", {}).get("oid")
        if local.is_file():
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            if expected is None or digest == expected:
                return {"path": relative.as_posix(), "sha256": digest}
        local.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                response = requests.get(remote, timeout=90)
                response.raise_for_status()
                content = response.content
                digest = hashlib.sha256(content).hexdigest()
                if expected and digest != expected:
                    raise ValueError(f"Checksum mismatch: {relative}")
                local.write_bytes(content)
                return {"path": relative.as_posix(), "sha256": digest}
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("Download failed")

    manifest = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, item in enumerate(executor.map(fetch, entries), start=1):
            manifest.append(item)
            if index % 100 == 0:
                print(f"Downloaded/verified {index}/{len(entries)} files", flush=True)
    original = yaml.safe_load((args.output / "data/data.yaml").read_text(encoding="utf-8"))
    names = original["names"]
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names != ["ball", "goalkeeper", "player", "referee"]:
        raise ValueError(f"Unexpected class order: {names}")
    configuration = {"path": str((args.output / "data").resolve()), "train": "train/images",
                     "val": "valid/images", "test": "test/images", "names": names}
    (args.output / "data.yaml").write_text(yaml.safe_dump(configuration), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({"repository": REPO, "revision": revision,
        "license": "CC BY 4.0", "source": "https://huggingface.co/datasets/" + REPO,
        "files": manifest}, indent=2), encoding="utf-8")
    print(f"Ready: {args.output / 'data.yaml'} ({len(manifest)} files)")


if __name__ == "__main__":
    main()
