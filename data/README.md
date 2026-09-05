# Soccer dataset and footage

The actual dataset used is [martinjolif/football-player-detection](https://huggingface.co/datasets/martinjolif/football-player-detection),
a public mirror attributed to [Football project on Roboflow Universe](https://universe.roboflow.com/football-project-pifbc/football-players-detection-3zvbc-yyhdl).
License: **CC BY 4.0**, as declared by the dataset publisher.
Credit: Football project, Roboflow Universe, and mirror maintainer Martin Jolif.

Verified revision: `e8b8cea002692efd74c945fcdad63e729adc5671`.
Actual splits: **298 train / 49 validation / 25 test** images.
Class indices: **0 ball, 1 goalkeeper, 2 player, 3 referee**.

Download with `python scripts/download_dataset.py`. It records the immutable
revision, per-file SHA256 hashes and source in `data/soccer/manifest.json`, checks
image LFS hashes and creates `data/soccer/data.yaml` with a local absolute path.
That machine-specific generated YAML is ignored by Git. Resume skips existing
verified image files. Label files are validated before training/evaluation.

No exact cross-split image hash duplicates were found. This does not establish
absence of near-duplicate frames, match overlap, or leakage from the third-party
model's training set. Results should be interpreted as in-domain checks.

The bundled local match is `2e57b9_0.mp4`, obtained through the link in
[Roboflow's soccer setup script](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh).
Their [soccer example](https://github.com/roboflow/sports/tree/main/examples/soccer)
identifies the original source as the DFL Bundesliga Data Shootout. The example
video is third-party footage, not project-owned media. Its source URL and checksum
are recorded in `samples/input/match.mp4.source.json`; it stays out of Git along
with derived video/GIF previews. Do not infer media redistribution rights from
the repository's code license.
