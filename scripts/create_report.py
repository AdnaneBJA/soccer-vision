"""Build an interactive local report and optional GIF from an existing completed run."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
from PIL import Image
from soccer_vision.video import VideoReader
from soccer_vision.visualization.report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    def read(name):
        return json.loads((args.artifacts / name).read_text(encoding="utf-8"))
    if read("status.json")["status"] != "complete":
        raise ValueError("Cannot report an incomplete run")
    write_report(args.artifacts, args.video, read("player_stats.json"), read("team_stats.json"),
                 read("benchmark.json"), read("config.json")["heatmaps"])
    if args.preview:
        images = []
        with VideoReader(args.video) as reader:
            stride = max(1, round(reader.fps / 5))
            for index, frame in enumerate(reader.frames()):
                if index >= round(reader.fps * 5):
                    break
                if index % stride == 0:
                    height = round(960 * reader.height / reader.width)
                    resized = cv2.resize(frame, (960, height))
                    images.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        if not images:
            raise ValueError("Video has no frames")
        images[0].save(args.artifacts / "preview.gif", save_all=True, append_images=images[1:],
                       duration=round(stride / reader.fps * 1000), loop=0)
        images[len(images) // 2].save(args.artifacts / "preview.jpg")
    print(args.artifacts / "report.html")


if __name__ == "__main__":
    main()
