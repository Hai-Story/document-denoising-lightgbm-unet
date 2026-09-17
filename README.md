# Document Denoising with U-Net

This branch contains a compact U-Net solution for restoring degraded grayscale document scans. It learns a direct mapping from noisy image patches to clean pixels, validates on complete held-out images, and produces competition-ready pixel predictions.

## Branches

- `main`: multiscale image features with LightGBM regression
- `unet`: patch-trained U-Net with full-image validation and flip test-time augmentation

## Project Sources

This is an independent solution to the following exercise and competition. It is not affiliated with their organizers.

- [AI Coding Gym — Denoising Dirty Documents](https://aicodinggym.com/challenges/mle/denoising-dirty-documents)
- [Kaggle — Denoising Dirty Documents](https://www.kaggle.com/competitions/denoising-dirty-documents)

The problem statement and dataset come from the sources above. This repository contains implementation code only and does not redistribute the dataset. Obtain the data from an authorized source and follow its terms of use.

## Method

The network uses a four-level encoder-decoder with skip connections. Each stage contains two convolution, batch normalization, and ReLU blocks. The implementation has approximately 7.76 million parameters with a base width of 32 channels.

Training uses the following configuration:

- 256 × 256 random patches
- Horizontal and vertical flip augmentation
- Batch size 8 and 200 optimization steps per epoch
- Adam optimizer with a starting learning rate of `1e-3`
- Cosine learning-rate schedule over 60 epochs
- Eight complete images reserved for validation
- Mean squared error training loss and full-image RMSE validation
- Apple Metal acceleration when available, with CPU fallback

The best checkpoint is selected by validation RMSE. Inference averages the original prediction with horizontal-flip and vertical-flip predictions, then quantizes the result to 8-bit grayscale before writing the submission file.

## Recorded Result

| Evaluation | RMSE (lower is better) |
| --- | ---: |
| Best full-image validation result | 0.01045 |

The recorded run trained on 107 images, validated on eight images, and completed 60 epochs in 6,427 seconds, or about 1 hour 47 minutes. The value above is a local validation result and is not a Kaggle leaderboard score. No external submission score is claimed for this branch.

The result comes from one fixed random split with seed `42`. It was not grouped by potentially shared source pages and was not repeated across multiple seeds, so it should not be treated as a robust estimate of performance on unrelated documents.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Arrange the downloaded data as follows:

```text
document-denoising-lightgbm/
├── train_unet.py
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
python train_unet.py
```

The script saves the best model as `unet_best.pt` and writes predictions to `submission.csv`. The CSV contains the columns `id,value`, where IDs follow the one-based `image_row_column` format expected by the competition.

The script retrains the network and overwrites the checkpoint and submission file. Runtime varies substantially by hardware; the recorded duration used Apple Metal acceleration.

## Repository Scope

Only source code, dependency declarations, documentation, and ignore rules are tracked. Datasets, submissions, model checkpoints, virtual environments, caches, compiled files, logs, and local tool configuration are excluded.

## Possible Improvements

- Group related source pages before splitting and repeat validation across several seeds.
- Add early stopping and save structured training metrics.
- Compare residual prediction, alternative normalization layers, and edge-aware losses.
- Evaluate larger training crops, tiled inference, and a validation-tuned blend with the LightGBM solution.

## Dataset Acknowledgment

According to the competition description, the dataset was created by RM.J. Castro-Bleda, S. España-Boquera, J. Pastor-Pellicer, and F. Zamora-Martinez, and was hosted by the UCI Machine Learning Repository. Consult the source pages for current citation requirements, including:

Bache, K. & Lichman, M. (2013). *UCI Machine Learning Repository*. Irvine, CA: University of California, School of Information and Computer Science.
