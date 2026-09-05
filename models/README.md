# Model checkpoints

No model is committed. Detection defaults to the official pretrained YOLO26 nano
COCO checkpoint, downloaded as `yolo26n.pt` on first use (ignored by Git).
Source: https://docs.ultralytics.com/models/yolo26/
Ultralytics distributes its code/models under AGPL-3.0 or an enterprise license:
https://www.ultralytics.com/license

Store fine-tuned local weights in `checkpoints/` (ignored by Git).
Document each checkpoint's source, license, architecture and class mapping when
detection is integrated. Generic person detections cannot distinguish soccer
players, goalkeepers and referees without additional training or classification.
