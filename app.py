import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

import config
from inventory import Inventory

log = logging.getLogger("inventory")


def setup_logging():
    if log.handlers:
        return
    config.LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOG_DIR / "scans.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return ""


def confirm(result, auto_yes):
    if auto_yes:
        return result["label"], "confirmed"
    answer = ask(f"Confirm '{result['label']}'? [y]es / [n]o: ")
    if answer in ("y", "yes"):
        return result["label"], "confirmed"
    if answer not in ("n", "no"):
        return None, "declined"
    others = result["top3"][1:]
    print("Other candidates:")
    for i, (label, p) in enumerate(others, 1):
        print(f"  {i}. {label} ({p:.2f})")
    pick = ask("Pick a number to correct the label, or press Enter to cancel: ")
    if pick.isdigit() and 1 <= int(pick) <= len(others):
        return others[int(pick) - 1][0], "corrected"
    return None, "declined"


def handle_scan(predictor, inv, image, source, qty, remove, auto_yes):
    result = predictor.predict(image)
    conf = result["confidence"]
    top = ", ".join(f"{label} {p:.2f}" for label, p in result["top3"])

    if not result["accepted"]:
        log.warning(
            f"FAILED  | {source} | not recognized, best guess {result['label']} "
            f"(conf {conf:.2f} < threshold {predictor.threshold:.2f})"
        )
        inv.log_scan(source, result["label"], conf, "rejected_low_conf")
        return

    log.info(f"PREDICT | {source} | {result['label']} (conf {conf:.2f}) | top3: {top}")
    final, status = confirm(result, auto_yes)
    if final is None:
        log.info(f"DECLINED| {source} | no stock change")
        inv.log_scan(source, result["label"], conf, "declined")
        return

    delta = -qty if remove else qty
    try:
        before, after = inv.change_stock(final, delta)
    except ValueError as err:
        log.error(f"NOSTOCK | {source} | {err}")
        inv.log_scan(source, result["label"], conf, "error", final, 0)
        return
    inv.log_scan(source, result["label"], conf, status, final, delta)
    log.info(f"STOCK   | {final}: {before} -> {after} ({delta:+d}) [{status}]")


def print_stats(inv):
    s = inv.stats()
    print("\n=== Inventory ===")
    print(f"Distinct products in stock: {s['products']}")
    print(f"Total units in stock:       {s['units']}")
    if s["stock"]:
        width = max(len(p) for p, _ in s["stock"])
        for product, qty in s["stock"]:
            print(f"  {product:<{width}}  {qty}")
    print("\n=== Scans ===")
    print(f"Total scans:               {s['scans_total']}")
    print(f"Recognized and confirmed:  {s['scans']['confirmed']}")
    print(f"Corrected by user:         {s['scans']['corrected']}")
    print(f"Declined by user:          {s['scans']['declined']}")
    print(f"Failed (low confidence):   {s['scans']['rejected_low_conf']}")
    print(f"Errors:                    {s['scans']['error']}")
    print(f"Success rate:              {s['success_rate']:.1%}\n")


def print_log(inv, n):
    for ts, source, predicted, conf, status, final, change in reversed(inv.recent_scans(n)):
        print(f"{ts} | {status:<17} | {predicted} ({conf:.2f}) -> {final or '-'} | change {change:+d} | {source}")


def run_camera(predictor, inv, args):
    import cv2

    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        log.error(f"cannot open camera {args.cam_index}")
        return
    config.CAPTURE_DIR.mkdir(exist_ok=True)
    print("Camera ready. SPACE = scan, Q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            log.error("camera frame read failed")
            break
        cv2.imshow("Inventory camera (SPACE scan, Q quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = config.CAPTURE_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(path), frame)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            handle_scan(predictor, inv, image, path.name, args.qty, args.remove, args.yes)
        elif key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Camera-based inventory tracker")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("scan", "camera"):
        p = sub.add_parser(name)
        if name == "scan":
            p.add_argument("image", type=Path)
        else:
            p.add_argument("--cam-index", type=int, default=0)
        p.add_argument("--qty", type=int, default=1)
        p.add_argument("--remove", action="store_true", help="subtract from stock instead of adding")
        p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
        p.add_argument("--threshold", type=float, default=config.CONF_THRESHOLD)

    sub.add_parser("stats")
    p_log = sub.add_parser("log")
    p_log.add_argument("--n", type=int, default=20)
    p_reset = sub.add_parser("reset")
    p_reset.add_argument("--force", action="store_true")

    args = parser.parse_args()
    setup_logging()
    inv = Inventory()

    if args.command == "stats":
        print_stats(inv)
    elif args.command == "log":
        print_log(inv, args.n)
    elif args.command == "reset":
        if args.force:
            inv.reset()
            log.info("inventory and scan history cleared")
        else:
            print("Add --force to clear all stock and scan history.")
    else:
        from predictor import Predictor

        predictor = Predictor(threshold=args.threshold)
        log.info(f"model loaded: {config.ACTIVE_MODEL['arch']} / {config.ACTIVE_MODEL['weights']}")
        if args.command == "scan":
            if not args.image.exists():
                log.error(f"image not found: {args.image}")
                sys.exit(1)
            with Image.open(args.image) as img:
                handle_scan(predictor, inv, img, args.image.name, args.qty, args.remove, args.yes)
        else:
            run_camera(predictor, inv, args)


if __name__ == "__main__":
    main()
