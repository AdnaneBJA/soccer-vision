"""Verify saved video timing, track records and accounting for a completed analysis run."""

import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_vision.video import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    def read(filename):
        return json.loads((args.artifacts / filename).read_text(encoding="utf-8"))
    benchmark, status, config, teams = [read(name) for name in ("benchmark.json", "status.json", "config.json", "team_stats.json")]
    if status["status"] != "complete":
        raise ValueError("Run is not complete")
    with VideoReader(args.video) as reader:
        count = sum(1 for _ in reader.frames())
        metadata = benchmark["video"]
        if count != benchmark["processed_frames"] or (reader.width, reader.height) != (metadata["width"], metadata["height"]) or abs(reader.fps - metadata["fps"]) > .001:
            raise ValueError("Output video count, dimensions or FPS do not match the report")
        duration = count / reader.fps
    identities = set()
    records = 0
    with (args.artifacts / "tracks.csv").open(newline="", encoding="utf-8") as stream:
        seen = set()
        for row in csv.DictReader(stream):
            frame, identity = int(row["frame"]), int(row["track_id"])
            if not 0 <= frame < count or (frame, identity) in seen:
                raise ValueError("Invalid frame or duplicate track record")
            seen.add((frame, identity))
            identities.add(identity)
            records += 1
    if config["tracking"] and teams["controlled_frames"] + teams["unknown_frames"] != count:
        raise ValueError("Possession accounting does not cover every frame")
    if len(identities) != benchmark["unique_tracks"]:
        raise ValueError("Unique track count does not match benchmark")
    if not (args.artifacts / "report.html").is_file():
        raise ValueError("HTML report missing")
    print(json.dumps({"verified": True, "frames": count, "duration_s": duration,
                      "track_records": records, "unique_tracks": len(identities)}, indent=2))


if __name__ == "__main__":
    main()
