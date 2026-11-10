# Camera-based Grocery Inventory Tracker

A camera (or any image file) looks at a product, a CNN names it with a confidence score, a human confirms, and the stock count in a SQLite database changes. Every scan is logged to the terminal and to the database, including failures. Stock survives restarts.

```
image / camera frame -> model prediction + confidence -> confirmation -> SQLite stock update -> log
```

Author: Md Sifatullah Sheikh

---

## 1. Quick start

Python 3.10 or newer.

```bash
git clone <private-repo-url>
cd grocery-inventory
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The trained weights (`models/best_cnn_v3.pth`) and `models/classes.json` are committed, so the app runs without training.

```bash
python app.py scan path/to/image.jpg          # predict, ask for confirmation, update stock
python app.py scan path/to/image.jpg --yes    # skip the confirmation prompt
python app.py scan path/to/image.jpg --remove # subtract from stock (a sale)
python app.py camera                          # live webcam: SPACE = scan, Q = quit
python app.py stats                           # products, quantities, scan success/failure counts
python app.py log --n 20                      # last 20 scans
```

Example session:

```
PREDICT | apple.jpg | 00_Golden-Delicious (conf 0.91) | top3: ...
Confirm '00_Golden-Delicious'? [y]es / [n]o: y
STOCK   | 00_Golden-Delicious: 0 -> 1 (+1) [confirmed]
```

If the confidence is below the threshold the item is rejected and nothing changes:

```
FAILED  | table.jpg | not recognized, best guess 12_Leek (conf 0.31 < threshold 0.50)
```

Answering `n` shows the next two most likely classes so the user can correct the label.

## 2. Project layout

| File | Purpose |
|---|---|
| `config.py` | Paths, active model, confidence threshold |
| `model.py` | `SimpleCNN` (from scratch) and pretrained options (ResNet-18, EfficientNet-B0) |
| `train.py` | Training script (AdamW, OneCycle, label smoothing, CutMix) |
| `evaluate.py` | Metric table, per-class report, confusion matrix, size, CPU latency, threshold sweep |
| `predictor.py` | Loads the model once, returns label, confidence, top-3, accepted/rejected |
| `inventory.py` | SQLite stock table and scan log |
| `app.py` | Terminal app: scan, camera, stats, log |
| `api.py`, `benchmark_api.py` | Bonus: FastAPI service and latency benchmark |

## 3. Reproducing the results

Dataset: the 60 classes of the Grocery Store Dataset that appear in train, val and test (see section 5). Place it as `data/d1-clean/{train,val,test}/<class>/<image>.jpg`.

```bash
python train.py --data data/d1-clean --arch simplecnn                 # from scratch, 100 epochs
python train.py --data data/d1-clean --arch efficientnet_b0 --pretrained   # fine-tuned, 30 epochs
python evaluate.py --data data/d1-clean --arch simplecnn --weights best_simplecnn.pth
```

`train.py` writes the weights to `models/`, the curves and history to `results/<name>/`, and prints the line to paste into `config.py` to make that model the active one. `evaluate.py` writes everything the report needs to `results/<name>/`: `metrics.md` (the table below), `per_class_report.csv`, `confusion_matrix.png`/`.csv`, `threshold_sweep.csv`, `metrics.json`.

Seed is fixed to 42 and cuDNN is set to deterministic. Data-loader augmentation workers are not fully deterministic, so a rerun can differ by about 1 point.

Hardware used for training: Kaggle, Tesla T4.

## 4. Design decisions

**Approach: image classifier.** The dataset provides one product per photo with a single class label, so a classifier fits the data directly. A detector would need bounding boxes the dataset does not have, and OCR would fail on loose fruit and vegetables, which make up most of the classes. The trade-off is that a classifier names the whole frame, so one product per image is assumed.

**Framework: PyTorch.** Simple `ImageFolder`/`DataLoader` pipeline and direct access to torchvision's pretrained models for the fine-tuning comparison.

**Storage: SQLite.** One file, no server, transactional (stock never half-updates), and queryable for the statistics. Two tables: `stock(product, quantity)` and `scans(ts, source, predicted, confidence, status, final_label, change)`. The database is written on every operation, so a shutdown cannot reset the count.

**Rejecting unknown items.** The prediction is accepted only if the top softmax probability is at least `CONF_THRESHOLD` (0.5 by default, set in `config.py`). Scans below it are logged as `rejected_low_conf` and never touch stock. `evaluate.py` writes `threshold_sweep.csv` (coverage and accuracy on accepted images per threshold) so the threshold can be chosen from data. Limitation: softmax confidence is only a proxy for "unknown", and a model trained with label smoothing and CutMix has lower peak probabilities than a plain one, so the threshold needs tuning per model.

**Human confirmation.** Stock changes only after the user confirms (or corrects) the prediction. `--yes` exists for scripted use.

**Scan statuses:** `confirmed`, `corrected` (model wrong, user fixed it), `declined`, `rejected_low_conf` (failed to recognise), `error` (for example removing more than is in stock). `stats` reports all of them and the success rate (confirmed / total).

**Model.** `SimpleCNN`: five blocks of two 3x3 convolutions with BatchNorm and ReLU, max-pooling after each block (32, 64, 128, 256, 512 channels), global average pooling, dropout, one linear layer. Global average pooling instead of a large dense layer keeps the parameter count and overfitting down.

**Training recipe.** AdamW (lr 3e-3, weight decay 5e-2), OneCycle schedule, label smoothing 0.1, CutMix (p=0.3), TrivialAugmentWide, random resized crop, random erasing, mixed precision, best checkpoint chosen by validation accuracy.

## 5. Dataset and categories

Dataset 1: Grocery Store Dataset (Klasson et al., WACV 2019, MIT licence). 81 fine classes in the original release. Only the 60 classes present in train, val and test were kept.

| Split | Images |
|---|---|
| Train | 1,996 |
| Val | 296 |
| Test | 1,889 |

Used categories: <!-- TODO: paste the 60 class names from models/classes.json -->

Carton items (juice, milk, yoghurt and similar packaged items; the task needs at least 10): <!-- TODO: list them; evaluate.py prints a count based on class names, confirm it by hand -->

## 6. Experiments

| Run | Setup | Epochs | Best val acc | Notes |
|---|---|---|---|---|
| v1 | 128 px, 4 conv blocks, basic augmentation, label smoothing | 60 | 0.747 | Train accuracy reached 0.99 while val stayed at 0.72 to 0.75: clear overfitting |
| v2 | 128 px, TrivialAugmentWide, random erasing, CutMix p=0.5, dropout 0.5 | 100 | 0.787 | Gap to train closed to about 4 points. Training time 9.8 min |
| v3 | 224 px, 5 conv blocks, dropout 0.3, CutMix p=0.3, milder crop and erasing | 100 | 0.807 | Training time 12.9 min. Clean accuracy train 1.000 / val 0.807 / test 0.746 |

![Training curves](results/best_cnn_v3/curves.png) <!-- TODO: add the curves image -->

Fine-tuned pretrained model: <!-- TODO: add EfficientNet-B0 run once trained -->

## 7. Results

Paste the table from `results/<name>/metrics.md` here for each trained model.

| Metric | Value | Must hit | Good |
|---|---|---|---|
| Test macro F1 | <!-- TODO --> | 0.80 | 0.92+ |
| Model file size | <!-- TODO --> | < 100 MB | < 30 MB |
| Median CPU prediction time (20 runs) | <!-- TODO --> | < 300 ms | < 80 ms |
| Parameters | <!-- TODO --> | | |
| Total training time | <!-- TODO --> | | |

Per-class precision, recall and F1: `results/<name>/per_class_report.csv`. Confusion matrix: `results/<name>/confusion_matrix.png`.

Known limitation: the from-scratch CNN reaches 0.746 test accuracy against 1.000 on the training images. With about 33 training images per class it memorises, and its test macro F1 is below the 0.80 pass line. This is why a pretrained model is trained and reported alongside it.

## 8. Error log and fixes

| Problem | Cause | Fix |
|---|---|---|
| Large train/val gap (0.99 vs 0.73) | About 33 images per class, model memorised | Stronger augmentation, CutMix, label smoothing, weight decay, global average pooling |
| Val accuracy jumping by 10+ points between epochs | Only 296 val images and a high learning rate mid-schedule | Smoothed curves for reading, test set used for the final number |
| Val and train both still rising at epoch 100 | Under-trained at 128 px | 224 px input, fifth conv block, lighter regularisation |
| Val 0.807 but test 0.746 | Best checkpoint was selected on a small val set | Report test as the main score; compare with the pretrained model |

<!-- TODO: add screenshots of: training curves, a successful scan, a rejected scan, the stats output, the camera window -->

## 9. Bonus: API

```bash
uvicorn api:app --port 8000
python benchmark_api.py path/to/image.jpg          # 20 requests, average and median latency
```

The model is loaded once at startup. Endpoints: `GET /health`, `POST /predict` (image upload), `POST /scan?qty=1&remove=false&confirmed=true` (updates stock only when `confirmed=true`), `GET /stats`.

Latency over 20 requests: <!-- TODO: average / median from benchmark_api.py -->

## 10. Notes

- `CONF_THRESHOLD` and the active model are set in `config.py`.
- `python app.py reset --force` clears stock and scan history.
- `captures/` holds frames saved by the camera mode; `logs/scans.log` mirrors the terminal log.
