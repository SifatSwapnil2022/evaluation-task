import hashlib
import io
from datetime import datetime

import streamlit as st
from PIL import Image, ImageOps

import config
from app import log, setup_logging
from inventory import Inventory
from predictor import Predictor

setup_logging()
inv = Inventory()

st.title("Grocery Inventory")

available = {}
for name, spec in config.MODELS.items():
    if (config.MODELS_DIR / spec["weights"]).exists():
        available[name] = spec
if not available:
    st.error("No model weights found in the models folder.")
    st.stop()


@st.cache_resource
def load_predictor(name):
    return Predictor(available[name])


model_name = st.sidebar.selectbox("Model", list(available))
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, config.CONF_THRESHOLD, 0.05)
action = st.sidebar.radio("Action", ["Add", "Remove"])
qty = st.sidebar.number_input("Quantity", min_value=1, max_value=100, value=1)
predictor = load_predictor(model_name)


def update_stock(scan):
    label = scan["result"]["label"]
    conf = scan["result"]["confidence"]
    delta = qty if action == "Add" else -qty
    try:
        before, after = inv.change_stock(label, delta)
    except ValueError as err:
        log.error(f"NOSTOCK | {scan['name']} | {err}")
        inv.log_scan(scan["name"], label, conf, "error", label, 0)
        return str(err)
    inv.log_scan(scan["name"], label, conf, "confirmed", label, delta)
    log.info(f"STOCK   | {label}: {before} -> {after} ({delta:+d}) [confirmed]")
    return f"{label}: {before} -> {after}"


source = st.radio("Image from", ["Upload", "Camera"], horizontal=True)
if source == "Upload":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a photo")

if file is None:
    st.write("Add an image to start.")
else:
    raw = file.getvalue()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    file_key = hashlib.md5(raw).hexdigest()
    st.image(image, width=300)

    if st.button("Scan"):
        name = file.name if source == "Upload" else f"camera_{datetime.now():%H%M%S}.jpg"
        result = predictor.predict(image)
        accepted = result["confidence"] >= threshold
        if accepted:
            log.info(f"PREDICT | {name} | {model_name} | {result['label']} (conf {result['confidence']:.2f})")
        else:
            log.warning(
                f"FAILED  | {name} | {model_name} | not recognized, best guess {result['label']} "
                f"(conf {result['confidence']:.2f} < threshold {threshold:.2f})"
            )
            inv.log_scan(name, result["label"], result["confidence"], "rejected_low_conf")
        st.session_state["scan"] = {
            "key": file_key,
            "name": name,
            "model": model_name,
            "threshold": threshold,
            "result": result,
            "accepted": accepted,
            "done": not accepted,
            "message": None,
        }

    scan = st.session_state.get("scan")
    if scan is not None and scan["key"] == file_key:
        result = scan["result"]
        st.subheader("Result")
        st.write(f"Model: {scan['model']}")
        st.write(f"Prediction: **{result['label']}**")
        st.write(f"Confidence: {result['confidence']:.1%}")
        for label, p in result["top3"]:
            st.write(f"- {label}: {p:.1%}")

        if not scan["accepted"]:
            st.error(f"Not recognized (confidence is below {scan['threshold']:.2f}). Stock was not changed.")
        elif scan["done"]:
            st.success(scan["message"])
        else:
            col1, col2 = st.columns(2)
            if col1.button("Confirm"):
                scan["message"] = update_stock(scan)
                scan["done"] = True
                st.rerun()
            if col2.button("Decline"):
                log.info(f"DECLINED| {scan['name']} | no stock change")
                inv.log_scan(scan["name"], result["label"], result["confidence"], "declined")
                scan["message"] = "Declined. Stock was not changed."
                scan["done"] = True
                st.rerun()

st.divider()
st.header("Stock")
stats = inv.stats()
st.write(f"Products in stock: {stats['products']}  |  Total units: {stats['units']}")
if stats["stock"]:
    st.table({
        "Product": [p for p, q in stats["stock"]],
        "Quantity": [q for p, q in stats["stock"]],
    })

st.header("Scans")
s = stats["scans"]
st.write(
    f"Total: {stats['scans_total']}  |  Confirmed: {s['confirmed']}  |  Declined: {s['declined']}  |  "
    f"Failed (low confidence): {s['rejected_low_conf']}  |  Success rate: {stats['success_rate']:.1%}"
)
rows = inv.recent_scans(10)
if rows:
    with st.expander("Recent scans"):
        st.table({
            "Time": [r[0] for r in rows],
            "Image": [r[1] for r in rows],
            "Predicted": [r[2] for r in rows],
            "Confidence": [round(r[3], 2) for r in rows],
            "Status": [r[4] for r in rows],
        })