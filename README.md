# Document Denoising with LightGBM

This repository contains a document image denoising solution based on multiscale image features and LightGBM pixel regression. The model restores text and paper backgrounds from degraded grayscale scans and predicts pixel intensities in the `[0, 1]` range.

## Branches

- `main`: multiscale image features with LightGBM regression
- `unet`: patch-trained U-Net with full-image validation and flip test-time augmentation

## Project Sources

This is an independent solution to the following exercise and competition. It is not affiliated with their organizers.

- [AI Coding Gym — Denoising Dirty Documents](https://aicodinggym.com/challenges/mle/denoising-dirty-documents)
- [Kaggle — Denoising Dirty Documents](https://www.kaggle.com/competitions/denoising-dirty-documents)

The problem statement and dataset come from the sources above. This repository contains implementation code only and does not redistribute the dataset. Obtain the data from an authorized source and follow its terms of use.

## Method

1. Extract neighboring pixels, Gaussian features at several scales, local medians, minima, and maxima.
2. Estimate the paper background with local maxima, smoothing, and morphological closing.
3. Derive normalized brightness, background difference, and local variance features.
4. Sample pixels from paired noisy and clean training images and fit a LightGBM regressor.
5. Select the number of boosting rounds on an image-level holdout, refit on all labeled images, and predict the test pixels.

This is a supervised feature-engineering approach and does not use pretrained models.

## Results

| Evaluation | RMSE (lower is better) |
| --- | ---: |
| Noisy pixels on the validation sample | 0.157925 |
| Clipped LightGBM validation predictions | 0.015319 |
| AI Coding Gym submission | 0.01416 |

Results were recorded on September 12, 2026. The submission score is from AI Coding Gym and is not a Kaggle leaderboard score.

The available data contained 115 labeled images and 29 test images. The validation split was stratified by image height and contained 95 training images and 20 validation images. Training sampled 8,000 pixels per image, while validation sampled 25,000 pixels per image. The local RMSE above is therefore a sampled-pixel metric rather than a full-image metric. Final training used all 115 labeled images, 8,000 sampled pixels per image, seed `20260912`, and 1,599 boosting rounds.

The split was not grouped by potentially shared clean source pages and was not repeated across multiple seeds. The validation estimate may therefore be optimistic, and library or platform differences may affect reproducibility.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. On macOS, LightGBM also requires an OpenMP runtime such as `libomp.dylib` that can be found by the dynamic linker. The runtime is not included in this repository.

Arrange the downloaded data as follows:

```text
document-denoising-lightgbm/
├── solve.py
├── requirements.txt
└── data/
    ├── train/
    ├── train_cleaned/
    ├── test/
    └── sampleSubmission.csv
```

Files with the same name in `train/` and `train_cleaned/` must form a noisy-clean pair.

## Run

```bash
python solve.py
```

The script writes `outputs/predictions.csv` with the columns `id,value`. Pixel IDs use the format `image_row_column`, with one-based row and column indices. Test images are processed in lexicographic filename order, and pixels are written in row-major order.

For the recorded submission, all 5,789,880 pixel IDs were checked against the provided sample submission in exact row order, and all predicted values were within `[0, 1]`. The script does not automatically submit results or perform this full CSV comparison, so verify the output again when using a different dataset release.

Running the script retrains the model and overwrites the prediction file. Trained model weights are not saved separately.

## Repository Scope

Only source code, dependency declarations, documentation, and ignore rules are tracked. Datasets, prediction files, model weights, virtual environments, caches, compiled files, logs, and local tool configuration are excluded.

## Possible Improvements

- Group related source pages before splitting and add full-image, repeated validation.
- Measure errors separately on text interiors, stroke edges, and paper backgrounds.
- Compare sampling strategies, background estimation windows, and LightGBM parameters.
- Evaluate ensembling with the U-Net implementation on the `unet` branch.

## Dataset Acknowledgment

According to the competition description, the dataset was created by RM.J. Castro-Bleda, S. España-Boquera, J. Pastor-Pellicer, and F. Zamora-Martinez, and was hosted by the UCI Machine Learning Repository. Consult the source pages for current citation requirements, including:

Bache, K. & Lichman, M. (2013). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science.
