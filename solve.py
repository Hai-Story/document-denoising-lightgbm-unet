"""Train a pixel-wise document restorer and write the competition submission.

All validation splits are by image, so neighboring pixels never cross the split.
Run from the repository root: .venv/bin/python solve.py
On macOS, LightGBM also needs an OpenMP runtime (libomp.dylib).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter, minimum_filter


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 20260912


def read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def features(image: np.ndarray) -> np.ndarray:
    """Multiscale context, smooth paper estimate, and local stroke structure."""
    x = image.astype(np.float32)
    h, w = x.shape
    padded = np.pad(x, 3, mode="reflect")
    planes = [x]
    for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0),
                   (-1, -1), (-1, 1), (1, -1), (1, 1),
                   (0, -2), (0, 2), (-2, 0), (2, 0),
                   (0, -3), (0, 3), (-3, 0), (3, 0)):
        planes.append(padded[3 + dy:3 + dy + h, 3 + dx:3 + dx + w])

    for sigma in (0.8, 1.6, 3, 8, 20, 45):
        planes.append(gaussian_filter(x, sigma))
    for size in (3, 5):
        planes.extend((median_filter(x, size), minimum_filter(x, size),
                       maximum_filter(x, size)))
    # A wide grayscale closing fills fine dark text but retains slow shadows.
    paper = maximum_filter(x, size=21)
    paper = gaussian_filter(paper, 8)
    paper_wide = gaussian_filter(maximum_filter(x, size=51), 16)
    planes.extend((paper, paper_wide, x / np.maximum(paper, 0.05),
                   x - paper, x / np.maximum(paper_wide, 0.05),
                   gaussian_filter(x * x, 3) - gaussian_filter(x, 3) ** 2))
    # A blur after morphological closing gives a second robust background cue.
    closed = cv2.morphologyEx(x, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))
    planes.append(gaussian_filter(closed, 8))
    return np.stack(planes, axis=-1).reshape(-1, len(planes))


def samples(paths: list[Path], count: int, rng: np.random.Generator):
    xs, ys = [], []
    for k, path in enumerate(paths, 1):
        image = read_image(path)
        target = read_image(DATA / "train_cleaned" / path.name)
        assert target.shape == image.shape
        indices = rng.choice(image.size, min(count, image.size), replace=False)
        xs.append(features(image)[indices])
        ys.append(target.ravel()[indices])
        if k % 25 == 0 or k == len(paths):
            print(f"Prepared {k}/{len(paths)} images", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def main():
    paths = sorted((DATA / "train").glob("*.png"))
    tests = sorted((DATA / "test").glob("*.png"))
    assert paths and tests, "Unzip data/denoising-dirty-documents.zip into data/ first"
    rng = np.random.default_rng(SEED)
    # Both page sizes occur in the held-out images.
    by_height = {}
    for path in paths:
        by_height.setdefault(Image.open(path).size[1], []).append(path)
    validation = []
    for group in by_height.values():
        validation.extend(rng.choice(group, size=max(2, round(len(group) * .17)),
                                     replace=False).tolist())
    holdout = set(validation)
    train = [p for p in paths if p not in holdout]
    print(f"Images: {len(train)} train, {len(validation)} validation, {len(tests)} test", flush=True)

    x_train, y_train = samples(train, 8000, rng)
    x_val, y_val = samples(validation, 25000, rng)
    baseline = np.sqrt(np.mean((x_val[:, 0] - y_val) ** 2))
    print(f"Noisy input RMSE: {baseline:.6f}", flush=True)
    params = dict(objective="regression", metric="rmse", learning_rate=0.06,
                  num_leaves=63, min_data_in_leaf=150, feature_fraction=0.88,
                  verbosity=-1, num_threads=6, seed=SEED)
    model = lgb.train(params, lgb.Dataset(x_train, label=y_train),
                      num_boost_round=1600,
                      valid_sets=[lgb.Dataset(x_val, label=y_val)],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])
    prediction = np.clip(model.predict(x_val, num_threads=6), 0, 1)
    rmse = np.sqrt(np.mean((prediction - y_val) ** 2))
    print(f"Holdout RMSE: {rmse:.6f}; best iteration {model.best_iteration}", flush=True)

    # Refit on every labeled image after model selection.
    x_all, y_all = samples(paths, 8000, np.random.default_rng(SEED + 1))
    final = lgb.train(params, lgb.Dataset(x_all, label=y_all),
                      num_boost_round=model.best_iteration)
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    result = output_dir / "predictions.csv"
    with result.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "value"))
        for k, path in enumerate(tests, 1):
            image = read_image(path)
            values = np.clip(final.predict(features(image), num_threads=6), 0, 1)
            for (r, c), value in zip(np.ndindex(image.shape), values):
                writer.writerow((f"{path.stem}_{r + 1}_{c + 1}", f"{value:.7f}"))
            print(f"Predicted {k}/{len(tests)} images", flush=True)
    print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
