"""Fetch publicly documented third-party soccer weights and example footage.

Assets remain local. Source and SHA256 are saved beside each download.
"""

import argparse
import hashlib
import json
from pathlib import Path
import requests

ASSETS = {
    "model": ("https://huggingface.co/gianpaj/football-players-detection-1/resolve/main/weights/best.pt",
              "models/checkpoints/best.pt", "AGPL-3.0", "https://huggingface.co/gianpaj/football-players-detection-1"),
    "video": ("https://drive.google.com/uc?export=download&id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf",
              "samples/input/match.mp4", "Original DFL footage; see source terms, not the code license",
              "https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh"),
}


def download(url: str, target: Path) -> str:
    if target.exists():
        raise FileExistsError(f"Already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Download returned a web page instead of an asset; download manually from the source")
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
    partial.replace(target)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("model", "video", "all"))
    args = parser.parse_args()
    for name in ASSETS if args.asset == "all" else [args.asset]:
        url, filename, license_name, source = ASSETS[name]
        target = Path(filename)
        checksum = download(url, target)
        target.with_suffix(target.suffix + ".source.json").write_text(json.dumps(
            {"url": url, "source": source, "license": license_name, "sha256": checksum}, indent=2), encoding="utf-8")
        print(f"Downloaded {target}, SHA256 {checksum}")


if __name__ == "__main__":
    main()
