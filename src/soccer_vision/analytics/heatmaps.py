"""Export image-coordinate occupancy maps without retaining whole videos."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_heatmap(grid: np.ndarray, path: Path, title: str, shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(grid, cmap="inferno", extent=(0, shape[1], shape[0], 0), aspect="auto")
    axis.set(title=title + " — image coordinates", xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    figure.colorbar(image, ax=axis, label="Observed frames")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
