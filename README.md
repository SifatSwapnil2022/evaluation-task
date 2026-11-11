# Grocery Inventory Tracker

Upload or capture a photo of a grocery item. A model names the product and gives a confidence score. You confirm it, and the stock count in a SQLite database changes. Every scan is logged, including failed ones. Stock survives restarts.

```
photo -> model (label + confidence) -> you confirm -> stock updated -> log
```

Author: Md Sifatullah Sheikh

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

## Run it on your own PC

You need Python 3.10 or newer and Git. No GPU is needed, the app runs on the CPU. No internet is needed after the install.

### 1. Clone

```bash
git clone https://github.com/SifatSwapnil2022/evaluation-task.git
cd evaluation-task
```

Go into the folder that contains `app.py` and `streamlit_app.py` (if there is an inner folder with the same name, `cd` into it).

### 2. Create an environment and install

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

What `pip install -r requirements.txt` installs:

| Package | Used for | Needed to run the web app? |
|---|---|---|
| torch, torchvision | The models | Yes |
| streamlit | The web app | Yes |
| numpy, pillow, pandas | Images and tables | Yes |
| opencv-python | Webcam in `app.py camera` | Only for the terminal camera |
| scikit-learn, matplotlib | `evaluate.py` (metrics, confusion matrix) | No |
| fastapi, uvicorn, python-multipart, requests, httpx | Optional API and its speed test | No |

### 3. Download the model weights

The weights are not in this repo (they are too big for git). Get them from the **Output** tab of the two Kaggle notebooks, then put them in the `models/` folder:

| Model | Notebook | File to download | Save it as |
|---|---|---|---|
| EfficientNet-B0 (main) | [efficientnet-b0-finetune](https://www.kaggle.com/code/xxxguruji/efficientnet-b0-finetune) | `best_effnet_b0_es.pth` | `models/best_effnet_b0_es.pth` |
| CNN from scratch | [last-day-v1-cnn](https://www.kaggle.com/code/mdsifatullahsheikh/last-day-v1-cnn) | `best_cnn_v3.pth` | `models/final_best_cnn_v3.pth`  |

```
models/
  classes.json                 (already in the repo)
  best_effnet_b0_es.pth
  final_best_cnn_v3.pth
```

The names must match `MODELS` in `config.py`. The app lists every model whose file exists, so one file is enough to start.

### 4. Start the web app

```bash
python -m streamlit run streamlit_app.py
```

It opens in the browser (http://localhost:8501). Pick a model, upload an image or use the camera, press Scan, then Confirm or Decline. The page shows the stock table and the scan statistics. Press Ctrl+C in the terminal to stop it.

Do not have a test image? Use any photo of one of the 60 products, or take one from the [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset).

### Terminal app (optional)

```bash
python app.py scan path/to/image.jpg
python app.py scan path/to/image.jpg --yes      # skip the confirmation question
python app.py scan path/to/image.jpg --remove   # a sale: subtract from stock
python app.py camera                            # webcam: SPACE = scan, Q = quit
python app.py stats
python app.py log --n 20
```

### If something goes wrong

| Problem | Fix |
|---|---|
| `weights not found` or the dropdown is empty | The `.pth` file is missing from `models/`, or its name does not match `config.py` |
| `streamlit is not recognized` | Use `python -m streamlit run streamlit_app.py` with the environment activated |
| `No module named ...` | Activate `.venv`, then run `pip install -r requirements.txt` again |
| `can't open file 'app.py'` | You are in the wrong folder. `cd` into the one that contains `app.py` |
| Camera does not open | Use Upload instead. For `app.py camera` try `--cam-index 1` |
| Most scans fail ("low confidence") | Lower the threshold slider in the web app (default 0.5) |

## How it works

- **Low confidence:** if the top probability is below the threshold (0.5 by default, `config.py`), the scan is logged as `rejected_low_conf` and the stock does not change. The web app has a slider.
- **Confirmation:** stock changes only after Confirm. Decline logs the scan and changes nothing.
- **Storage:** SQLite, two tables: `stock` and `scans`. The database is created on the first run in `db/inventory.db` and saved after every change.
- **Scan statuses:** confirmed, corrected (terminal app only), declined, rejected_low_conf, error.
- **Logs:** each scan is printed in the terminal and saved in `logs/scans.log`.

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

- CNN: https://www.kaggle.com/code/mdsifatullahsheikh/last-day-v1-cnn
- EfficientNet-B0: https://www.kaggle.com/code/xxxguruji/efficientnet-b0-finetune

CNN: Training Curves

<img width="1241" height="436" alt="CNN_TrainingCurve" src="https://github.com/user-attachments/assets/4c1b26b4-63c7-4cbd-8f53-f626f095691f" />

EfficientNet-B0: Training Curves

<img width="1389" height="490" alt="EfficientNetB0_TrainingCurve" src="https://github.com/user-attachments/assets/d6334fab-69bf-4bf3-a8ba-131586d4f53e" />

Dataset: [Grocery Store Dataset](https://github.com/marcusklasson/GroceryStoreDataset), only the 60 classes that appear in train, val and test. Cleaned copy: https://www.kaggle.com/datasets/mdsifatullahsheikh/d1-clean/data. The dataset is not needed to run the app, only to retrain or to run `evaluate.py`.

## Limits

- One product per image.
- Softmax confidence is only a rough sign of "unknown".
- The scratch CNN memorises the training set (train 100%, test 74.6%) because there are about 33 images per class.
