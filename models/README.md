# Checkpoint provenance

`models/checkpoints/best.pt` is the downloaded reference checkpoint from
[gianpaj/football-players-detection-1](https://huggingface.co/gianpaj/football-players-detection-1).
It is a YOLOv8x soccer detector with label order ball, goalkeeper, player, referee.
The publisher declares **AGPL-3.0**. Credit belongs to its publisher and Ultralytics.
We evaluated this checkpoint; we did not perform its original training.

Downloaded SHA256:
`a35ca40ea9e728288b86b37f728afbe601dfd7ec58f30d4661900c2d9b308932`.
`python scripts/download_assets.py model` records source/checksum metadata locally.

`models/checkpoints/soccer_yolo26n_cpu3.pt` is our actual three-epoch fine-tuning
run from official pretrained `yolo26n.pt`, using the documented soccer dataset,
640-pixel inputs, batch size 8, AdamW, seed 42 and CPU execution. It remains an
undertrained baseline: test mAP@50 is approximately 0.183 versus 0.815 for the
reference checkpoint. The reference model remains the default for better soccer
role detection. Actual metrics and learning curves are in `reports/latest/`.

Generic `yolo26n.pt` is automatically downloaded by Ultralytics when requested.
It provides COCO person/sports-ball labels and cannot distinguish soccer roles
without training. Model weights and generated checkpoints are never committed.
