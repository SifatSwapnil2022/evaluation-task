# Grocery Inventory Tracker

Upload or capture a photo of a grocery item. A model names the product and gives a confidence score. You confirm it, and the stock count in a SQLite database changes. Every scan is logged, including failed ones. Stock survives restarts.

```
photo -> model (label + confidence) -> you confirm -> stock updated -> log
```


## Models

Two models, trained on 60 classes of the Grocery Store Dataset (train 1,996 / val 296 / test 1,889 images, 224 x 224).

| | CNN (from scratch) | EfficientNet-B0 (fine-tuned) |
|---|---|---|
| Test accuracy | 74.6% | 86.6% |
| Test macro F1 | 0.7525 | 0.8877 |
| Validation accuracy | 80.7% | 92.2% |
| Parameters | 4.74 M | 4.08 M |
| File size | 18.1 MB | 15.9 MB |
| CPU time per image (median) | 96.2 ms | 40.3 ms |
| Training time (Kaggle T4) | 12.9 min | 5.7 min |

CPU time was measured on the Kaggle CPU. EfficientNet-B0 is the main model.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate         
pip install -r requirements.txt
```

## Run

Web app:

```bash
python -m streamlit run streamlit_app.py
```

Pick a model, upload an image or use the camera, press Scan, then Confirm or Decline. The page shows the stock table, scan statistics.

Terminal app:

```bash
python app.py scan path/to/image.jpg
python app.py camera
python app.py stats
python app.py log --n 20
```

## How it works

- **Low confidence:** if the top probability is below the threshold (0.5 by default, `config.py`), the scan is logged as `rejected_low_conf` and the stock does not change. The web app has a slider.
- **Confirmation:** stock changes only after Confirm. Decline logs the scan and changes nothing.
- **Storage:** SQLite, two tables: `stock` and `scans`. Saved after every change.
- **Scan statuses:** confirmed, corrected (terminal app only), declined, rejected_low_conf, error.

## Files

| File | What it does |
|---|---|
| `config.py` | Model list, weights names, threshold |
| `model.py` | Model code and image transforms |
| `predictor.py` | Loads a model, returns label, confidence, top 3 |
| `inventory.py` | SQLite stock and scan log |
| `streamlit_app.py` | Web app |
| `app.py` | Terminal app |
| `train.py` | Training script |
| `evaluate.py` | Accuracy, macro F1, per-class report, confusion matrix, size, CPU speed |
| `api.py`, `benchmark_api.py` | FastAPI service and speed test |

## Training

Both models were trained in Kaggle notebooks:

- CNN: Training Curves :

 <img width="1241" height="436" alt="CNN_TrainingCurve" src="https://github.com/user-attachments/assets/4c1b26b4-63c7-4cbd-8f53-f626f095691f" />

  
- EfficientNet-B0 : Training Curves :

  <img width="1389" height="490" alt="EfficientNetB0_TrainingCurve" src="https://github.com/user-attachments/assets/d6334fab-69bf-4bf3-a8ba-131586d4f53e" />


Dataset: [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset), only the 60 classes that appear in train, val and test.

## Limits

- One product per image.
- Softmax confidence is only a rough sign of "unknown".
- The scratch CNN memorises the training set (train 100%, test 74.6%) because there are about 33 images per class.
