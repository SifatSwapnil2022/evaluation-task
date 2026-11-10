import argparse
import json
import random
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from model import build_model, eval_transform, train_transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRESETS = {
    "simplecnn": {"lr": 3e-3, "wd": 5e-2, "epochs": 100},
    "pretrained": {"lr": 1e-3, "wd": 1e-2, "epochs": 30},
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a grocery classifier")
    p.add_argument("--data", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--arch", default="simplecnn", choices=["simplecnn", "resnet18", "efficientnet_b0"])
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights")
    p.add_argument("--name", default=None, help="output name, default best_<arch>")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None)
    p.add_argument("--cutmix", type=float, default=0.3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    preset = PRESETS["simplecnn" if args.arch == "simplecnn" and not args.pretrained else "pretrained"]
    args.epochs = args.epochs or preset["epochs"]
    args.lr = args.lr or preset["lr"]
    args.wd = args.wd if args.wd is not None else preset["wd"]
    args.name = args.name or f"best_{args.arch}"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cutmix(x, y):
    lam = np.random.beta(1.0, 1.0)
    idx = torch.randperm(x.size(0), device=x.device)
    h, w = x.shape[2], x.shape[3]
    cut = np.sqrt(1 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
    return x, y, y[idx], lam


def run_epoch(model, loader, criterion, device, use_amp, train, optimizer=None, scheduler=None, scaler=None, cutmix_prob=0.0):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            use_cutmix = train and np.random.rand() < cutmix_prob
            if use_cutmix:
                x, y_a, y_b, lam = cutmix(x, y)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                if use_cutmix:
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def smooth(values, k=5):
    padded = np.pad(values, (k // 2, k // 2), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def plot_curves(history, path):
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "acc", "Accuracy (%)"), (axes[1], "loss", "Loss")]:
        scale = 100 if key == "acc" else 1
        tr = np.array(history[f"train_{key}"]) * scale
        va = np.array(history[f"val_{key}"]) * scale
        tr_s, va_s = smooth(tr), smooth(va)
        ax.fill_between(epochs, tr_s, va_s, color="gold", alpha=0.3, label="Train-val gap")
        ax.plot(epochs, tr_s, color="green", lw=2.5, label="Train")
        ax.plot(epochs, va_s, color="red", lw=2.5, label="Validation")
        if key == "acc":
            best = va.argmax()
            ax.scatter(best + 1, va[best], marker="*", s=200, color="red", edgecolor="black", zorder=5)
            ax.axhline(80, color="gray", ls="--", lw=1, label="80% target")
            ax.set_ylim(0, 100)
            ax.set_title(f"{title}  |  best val {va[best]:.1f}% (epoch {best + 1})", fontweight="bold")
            ax.legend(loc="upper left")
        else:
            ax.set_title(title, fontweight="bold")
            ax.legend(loc="upper right")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"device: {device} | arch: {args.arch} | pretrained: {args.pretrained}")

    train_ds = datasets.ImageFolder(f"{args.data}/train", transform=train_transform(args.img_size))
    val_ds = datasets.ImageFolder(f"{args.data}/val", transform=eval_transform(args.img_size))
    assert train_ds.classes == val_ds.classes
    classes = train_ds.classes
    config.MODELS_DIR.mkdir(exist_ok=True)
    (config.MODELS_DIR / "classes.json").write_text(json.dumps(classes, indent=2))

    loader_args = {"batch_size": args.batch, "num_workers": args.workers, "pin_memory": use_amp, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = build_model(args.arch, len(classes), pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"classes: {len(classes)} | train/val: {len(train_ds)}/{len(val_ds)} | parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = config.RESULTS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, start = 0.0, time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device, use_amp, True, optimizer, scheduler, scaler, args.cutmix
        )
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp, False)
        for key, value in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[key].append(value)
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), config.MODELS_DIR / f"{args.name}.pth")
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")

    train_time = time.time() - start
    history.update({
        "train_time_min": train_time / 60,
        "best_val_acc": best_val,
        "parameters": n_params,
        "args": vars(args),
    })
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_curves(history, out_dir / "curves.png")

    print(f"\nbest val acc: {best_val:.3f} | training time: {train_time / 60:.1f} min")
    print(f"saved: models/{args.name}.pth, results/{args.name}/history.json, results/{args.name}/curves.png")
    print("to use it in the app, set in config.py:")
    print(f'ACTIVE_MODEL = {{"arch": "{args.arch}", "weights": "{args.name}.pth", "img_size": {args.img_size}}}')


if __name__ == "__main__":
    main()
