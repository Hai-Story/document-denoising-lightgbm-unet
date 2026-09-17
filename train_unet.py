"""U-Net denoiser for Kaggle 'Denoising Dirty Documents' (MLE-bench).

Trains a small U-Net on (train, train_cleaned) image pairs with patch
augmentation, monitors RMSE on a held-out validation split, then writes
predictions for the test images in the submission format:
    id,value   with id = {image}_{row}_{col} (1-indexed, row-major).
"""
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "submission.csv")

SEED = 42
PATCH = 256
EPOCHS = 60
BATCH = 8
LR = 1e-3
VAL_IMAGES = 8
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_pair(path_dirty, path_clean=None):
    x = np.asarray(Image.open(path_dirty), dtype=np.float32) / 255.0
    if path_clean is None:
        return x
    y = np.asarray(Image.open(path_clean), dtype=np.float32) / 255.0
    return x, y


# ---------------- model ----------------
class Conv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.d1 = Conv(1, base)
        self.d2 = Conv(base, base * 2)
        self.d3 = Conv(base * 2, base * 4)
        self.d4 = Conv(base * 4, base * 8)
        self.bottom = Conv(base * 8, base * 16)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.c4 = Conv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = Conv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = Conv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = Conv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph = (16 - h % 16) % 16
        pw = (16 - w % 16) % 16
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.d1(x)
        e2 = self.d2(self.pool(e1))
        e3 = self.d3(self.pool(e2))
        e4 = self.d4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = self.c4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.c3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.c2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.c1(torch.cat([self.u1(d2), e1], dim=1))
        y = self.out(d1)
        return y[..., :h, :w].clamp(0, 1)


# ---------------- data ----------------
def main():
    train_ids = sorted(f[:-4] for f in os.listdir(os.path.join(DATA, "train")) if f.endswith(".png"))
    test_ids = sorted(f[:-4] for f in os.listdir(os.path.join(DATA, "test")) if f.endswith(".png"))

    random.Random(SEED).shuffle(train_ids)
    val_ids = train_ids[:VAL_IMAGES]
    fit_ids = train_ids[VAL_IMAGES:]

    pairs = []
    for i in fit_ids:
        x, y = load_pair(
            os.path.join(DATA, "train", i + ".png"),
            os.path.join(DATA, "train_cleaned", i + ".png"),
        )
        pairs.append((i, x, y))
    print(f"fit={len(pairs)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    model = UNet(base=32).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M device={DEV}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    mse = nn.MSELoss()

    def sample_batch():
        xs, ys = [], []
        for _ in range(BATCH):
            _, x, y = random.choice(pairs)
            h, w = x.shape
            ph, pw = min(PATCH, h), min(PATCH, w)
            r = random.randint(0, h - ph)
            c = random.randint(0, w - pw)
            xp = x[r:r + ph, c:c + pw]
            yp = y[r:r + ph, c:c + pw]
            if random.random() < 0.5:
                xp, yp = xp[:, ::-1], yp[:, ::-1]
            if random.random() < 0.5:
                xp, yp = xp[::-1], yp[::-1]
            xs.append(xp.copy())
            ys.append(yp.copy())
        xt = torch.from_numpy(np.stack(xs)[:, None])
        yt = torch.from_numpy(np.stack(ys)[:, None])
        return xt.to(DEV), yt.to(DEV)

    steps_per_epoch = 200
    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run = 0.0
        for _ in range(steps_per_epoch):
            xt, yt = sample_batch()
            pred = model(xt)
            loss = mse(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()

        # validation: full-image RMSE
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in val_ids:
                x, y = load_pair(
                    os.path.join(DATA, "train", i + ".png"),
                    os.path.join(DATA, "train_cleaned", i + ".png"),
                )
                xt = torch.from_numpy(x)[None, None].to(DEV)
                yt = torch.from_numpy(y).to(DEV)
                pred = model(xt)[0, 0]
                tot += ((pred - yt) ** 2).mean().item() * y.size
                cnt += y.size
        val_rmse = (tot / cnt) ** 0.5
        marker = ""
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), os.path.join(BASE, "unet_best.pt"))
            marker = " *saved*"
        print(f"epoch {epoch:3d}/{EPOCHS} train_mse={run/steps_per_epoch:.5f} "
              f"val_rmse={val_rmse:.5f}{marker} ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- inference with flip TTA ----------------
    model.load_state_dict(torch.load(os.path.join(BASE, "unet_best.pt"), weights_only=True))
    model.eval()

    def predict(x):
        xt = torch.from_numpy(x)[None, None].to(DEV)
        with torch.no_grad():
            out = model(xt)[0, 0]
            out = (out + torch.flip(model(torch.flip(xt, dims=[3])), dims=[3])[0, 0]
                   + torch.flip(model(torch.flip(xt, dims=[2])), dims=[2])[0, 0]) / 3.0
        return out.clamp(0, 1).cpu().numpy()

    with open(OUT, "w") as f:
        f.write("id,value\n")
        for i in test_ids:
            x = load_pair(os.path.join(DATA, "test", i + ".png"))
            pred = predict(x)
            pred16 = (pred * 255).round().astype(np.uint8)  # match PNG quantization
            # write rows: id is 1-indexed image_row_col
            h, w = pred.shape
            vals = (pred16.astype(np.float32) / 255.0).ravel()
            lines = np.empty(h * w, dtype=object)
            idx = np.arange(h * w)
            r = (idx // w) + 1
            c = (idx % w) + 1
            f.write("\n".join(f"{i}_{rr}_{cc},{v:.6g}"
                              for rr, cc, v in zip(r.tolist(), c.tolist(), vals.tolist())))
            f.write("\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
