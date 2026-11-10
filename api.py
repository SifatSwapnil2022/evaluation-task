import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inventory import Inventory
from predictor import Predictor

state = {}


@asynccontextmanager
async def lifespan(app):
    state["predictor"] = Predictor()
    yield
    state.clear()


app = FastAPI(title="Grocery inventory API", lifespan=lifespan)


async def read_image(file):
    try:
        return Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="file is not a valid image")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = state["predictor"].predict(await read_image(file))
    result["top3"] = [{"label": label, "prob": p} for label, p in result["top3"]]
    return result


@app.post("/scan")
async def scan(file: UploadFile = File(...), qty: int = 1, remove: bool = False, confirmed: bool = False):
    result = state["predictor"].predict(await read_image(file))
    inv = Inventory()
    source = file.filename or "upload"
    if not result["accepted"]:
        inv.log_scan(source, result["label"], result["confidence"], "rejected_low_conf")
        return {"status": "rejected_low_conf", "label": result["label"], "confidence": result["confidence"]}
    if not confirmed:
        return {"status": "pending_confirmation", "label": result["label"], "confidence": result["confidence"]}
    delta = -qty if remove else qty
    try:
        before, after = inv.change_stock(result["label"], delta)
    except ValueError as err:
        inv.log_scan(source, result["label"], result["confidence"], "error", result["label"], 0)
        raise HTTPException(status_code=409, detail=str(err))
    inv.log_scan(source, result["label"], result["confidence"], "confirmed", result["label"], delta)
    return {"status": "confirmed", "label": result["label"], "before": before, "after": after}


@app.get("/stats")
def stats():
    return Inventory().stats()
