import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

import config
from model import eval_transform, load_trained

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model and write the report metrics")
    p.add_argument("--data", default=str(config.DEFAULT_DATA_DIR))
    p.add_argument("--arch", default=config.ACTIVE_MODEL["arch"])
    p.add_argument("--weights", default=config.ACTIVE_MODEL["weights"])
    p.add_argument("--img-size", type=int, default=config.ACTIVE_MODEL["img_size"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--train-time-min", type=float, default=None, help="use if training was done in a notebook")
    return p.parse_args()


@torch.no_grad()
def collect(model, loader, device):
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def median_cpu_latency(arch, weights, img_size, image_path, runs):
    model, _ = load_trained(arch, weights, "cpu")
    transform = eval_transform(img_size)
    times = []
    with torch.no_grad():
        for i in range(runs + 3):
            start = time.perf_counter()
            with Image.open(image_path) as img:
                x = transform(img.convert("RGB")).unsqueeze(0)
            torch.softmax(model(x), dim=1).argmax(1)
            if i >= 3:
                times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def grade(value, must, good, lower_is_better):
    ok = value < must if lower_is_better else value >= must
    best = value < good if lower_is_better else value >= good
    return "good" if best else ("pass" if ok else "FAIL")


def save_confusion(cm, classes, path):
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set, row-normalised)")
    fig.colorbar(im, fraction=0.03)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = Path(args.weights).stem
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, classes = load_trained(args.arch, args.weights, device)
    transform = eval_transform(args.img_size)
    results, test_probs, test_targets, test_ds = {}, None, None, None

    for split in ("train", "val", "test"):
        if not (Path(args.data) / split).exists():
            print(f"{split:5s} | folder not found, skipped")
            continue
        ds = datasets.ImageFolder(f"{args.data}/{split}", transform=transform)
        if ds.classes != classes:
            raise SystemExit(f"class folders in {split}/ do not match models/classes.json")
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
        probs, targets = collect(model, loader, device)
        preds = probs.argmax(1)
        results[split] = {
            "accuracy": float((preds == targets).mean()),
            "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "images": len(ds),
        }
        print(f"{split:5s} | acc {results[split]['accuracy']:.3f} | macro F1 {results[split]['macro_f1']:.3f} | n={len(ds)}")
        if split == "test":
            test_probs, test_targets, test_ds = probs, targets, ds

    if test_probs is None:
        raise SystemExit(f"no test folder found in {args.data}")
    preds = test_probs.argmax(1)
    report = classification_report(test_targets, preds, target_names=classes, output_dict=True, zero_division=0)
    with open(out_dir / "per_class_report.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for c in classes:
            r = report[c]
            writer.writerow([c, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", int(r["support"])])

    cm = confusion_matrix(test_targets, preds, labels=range(len(classes)))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion(cm, classes, out_dir / "confusion_matrix.png")

    conf = test_probs.max(1)
    correct = preds == test_targets
    sweep = []
    for t in np.arange(0.0, 0.951, 0.05):
        accepted = conf >= t
        sweep.append({
            "threshold": round(float(t), 2),
            "coverage": float(accepted.mean()),
            "accuracy_on_accepted": float(correct[accepted].mean()) if accepted.any() else 0.0,
        })
    with open(out_dir / "threshold_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = os.path.getsize(config.MODELS_DIR / args.weights) / (1024 ** 2)
    latency = median_cpu_latency(args.arch, args.weights, args.img_size, test_ds.samples[0][0], args.runs)

    train_time = args.train_time_min
    history_path = out_dir / "history.json"
    if train_time is None and history_path.exists():
        train_time = json.loads(history_path.read_text()).get("train_time_min")

    cartons = [c for c in classes if any(k in c.lower() for k in config.CARTON_KEYWORDS)]
    metrics = {
        "model": args.arch,
        "weights": args.weights,
        "splits": results,
        "test_macro_f1": results["test"]["macro_f1"],
        "model_size_mb": size_mb,
        "median_cpu_latency_ms": latency,
        "latency_runs": args.runs,
        "parameters": n_params,
        "train_time_min": train_time,
        "classes": len(classes),
        "carton_like_classes": len(cartons),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    f1 = results["test"]["macro_f1"]
    rows = [
        ("Test macro F1", f"{f1:.3f}", "0.80", "0.92+", grade(f1, 0.80, 0.92, False)),
        ("Model file size", f"{size_mb:.1f} MB", "< 100 MB", "< 30 MB", grade(size_mb, 100, 30, True)),
        (f"Median CPU prediction time ({args.runs} runs)", f"{latency:.1f} ms", "< 300 ms", "< 80 ms", grade(latency, 300, 80, True)),
    ]
    lines = [f"### {args.arch} ({args.weights})", "", "| Metric | Value | Must hit | Good | Status |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    lines += ["", "| Split | Accuracy | Macro F1 | Images |", "|---|---|---|---|"]
    lines += [f"| {s} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['images']} |" for s, r in results.items()]
    lines += [
        "",
        f"- Parameters: {n_params:,}",
        f"- Total training time: {f'{train_time:.1f} min' if train_time else 'not recorded'}",
        f"- Classes: {len(classes)} (carton-like by name: {len(cartons)})",
    ]
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n")

    worst = sorted(classes, key=lambda c: report[c]["f1-score"])[:10]
    print("\n".join(lines))
    print("\nweakest classes (test F1):")
    for c in worst:
        print(f"  {c:<45} f1 {report[c]['f1-score']:.2f}  recall {report[c]['recall']:.2f}")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()