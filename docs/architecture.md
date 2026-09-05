# Understanding SoccerVision

Start with `src/soccer_vision/pipeline.py`. Its frame loop follows the actual
processing order; the detector, tracker and analytics modules expose the data
passed between stages instead of hiding the whole application in a library call.

## Pixels to PyTorch predictions

OpenCV decodes one `uint8` array with shape `(height, width, 3)` in BGR order.
Ultralytics letterboxes it, converts BGR to RGB, normalizes pixel values and
constructs a batched `(N, 3, H, W)` tensor. It places the tensor and model on the
selected CPU/CUDA device. Inference runs without optimizer updates. During
training the loss produces gradients through backpropagation, and AdamW updates
the pretrained weights. The detection head must learn the four soccer labels.

Returned `xyxy` coordinates already refer to the original image, so do not divide
them by the model input size or apply a second resize. The adapter transfers
result tensors to CPU and exposes plain `Detection` dataclasses. This transfer
also synchronizes CUDA for the measured end-to-end prediction latency.

## Detections to identities

ByteTrack predicts human motion with a Kalman filter. It first associates strong
detections, then uses lower-confidence detections to recover existing tracks.
Dropping weak detections before tracking defeats the second pass, which is why
the default detector threshold is 0.1 and the first association threshold is 0.25.
We pin Ultralytics because the direct ByteTrack API has changed between versions.

Separate class trackers prevent cross-class matching. Public IDs are mapped
independently from library IDs and never reused after a scene reset. A player
misclassified as a goalkeeper can still acquire a new identity: this is an
explicit limitation of class separation. Broadcast cuts are detected using HSV
histogram distance; similar-looking cuts may be missed.

The ball uses a separate constant-velocity distance gate because tiny boxes
often do not overlap across frames. Predictions guide association only. The
CSV and possession logic receive observed detections, never imagined ball
locations during gaps. False balls and abrupt motion can still break tracks.

## Identity to team

The classifier samples the central upper torso, rejects obvious green grass,
and takes a median color feature. Deterministic two-means clustering establishes
anonymous Team 1 and Team 2 centers. Votes over a track's observed jersey colors
reduce flicker. Initial warm-up observations remain unknown; labels are not
retroactively written into already exported rows.

Referees and goalkeepers do not enter the jersey clustering. Goalkeepers remain
unassigned because their jerseys differ from outfield players. A green kit can
be mistaken for grass; similar uniforms, shadows and class mistakes remain
failure cases. Lab is the default; RGB and circular HSV are configurable.

## Analytics and coordinate systems

Track CSV rows contain bottom-center foot positions and optional field positions.
For each player, exponential smoothing reduces jitter; distance is accumulated
only between consecutive observed frames. Missing frames do not create straight
line jumps. Image-space speed is pixels/second and includes camera motion.
`observed_seconds` measures the time intervals used in motion statistics, not
total time on the pitch. Top speed is the maximum smoothed observed step speed.

Possession chooses the nearest classified outfield player's feet when the ball
is within the configured pixel threshold. Unknown frames stay in the report.
Team percentages divide by controlled frames only. They are not match-event
ground truth, and goalkeeper possession is unknown without a separate team rule.

Occupancy heatmaps count observed foot positions in a fixed image grid. They
aggregate screen locations, including different camera views. They are not pitch
coverage maps. Tracks CSV includes scene IDs for further scene-specific analysis.

## Fixed-camera calibration

Supply at least four non-collinear image points and corresponding meter-valued
field coordinates in JSON:

```json
{
  "camera": "fixed",
  "image_points": [[100, 100], [900, 100], [900, 600], [100, 600]],
  "field_points_m": [[0, 0], [105, 0], [105, 68], [0, 68]]
}
```

These coordinates illustrate the file format only. Replace them with measured
correspondences for your fixed camera; do not use them on the bundled broadcast.
Pass the file with `--calibration path/to/calibration.json`. Homography maps foot
positions to meters; the analytics exports distance in meters and speed in km/h.
Detected camera cuts invalidate calibration. Pan/tilt/zoom without cuts also
invalidates a static homography but cannot be reliably detected here. No real
calibrated soccer speed measurements are claimed by this project.

## Evaluation discipline

Keep train/validation/test splits distinct. Training validates class order,
normalized box coordinates, missing annotations and exact cross-split image
duplicates. Exact hashes cannot rule out adjacent near-duplicate frames. The
reference checkpoint's training footage may overlap the public mirror: current
scores demonstrate in-domain behavior, not independent generalization.

Threshold experiments use validation images and cached predictions. They report
fixed-threshold precision/recall with IoU matching; mAP is measured separately by
Ultralytics across confidence levels and IoU thresholds. Tracking evaluation
uses standard MOT metrics on explicitly supplied fully annotated frame ranges.
Known-answer synthetic tests verify the evaluator, but no real tracking ground
truth was available for the bundled video, so no real MOTA/IDF1 score is claimed.
