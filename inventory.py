import sqlite3
from datetime import datetime

import config

STATUSES = ("confirmed", "corrected", "declined", "rejected_low_conf", "error")


class Inventory:
    def __init__(self, path=config.DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS stock ("
            "product TEXT PRIMARY KEY, quantity INTEGER NOT NULL DEFAULT 0)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS scans ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, predicted TEXT, "
            "confidence REAL, status TEXT, final_label TEXT, change INTEGER)"
        )
        self.conn.commit()

    def quantity(self, product):
        row = self.conn.execute("SELECT quantity FROM stock WHERE product = ?", (product,)).fetchone()
        return row[0] if row else 0

    def change_stock(self, product, delta):
        before = self.quantity(product)
        after = before + delta
        if after < 0:
            raise ValueError(f"cannot remove {-delta} of {product}: only {before} in stock")
        self.conn.execute(
            "INSERT INTO stock (product, quantity) VALUES (?, ?) "
            "ON CONFLICT(product) DO UPDATE SET quantity = excluded.quantity",
            (product, after),
        )
        self.conn.commit()
        return before, after

    def log_scan(self, source, predicted, confidence, status, final_label=None, change=0):
        self.conn.execute(
            "INSERT INTO scans (ts, source, predicted, confidence, status, final_label, change) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), str(source), predicted, confidence, status, final_label, change),
        )
        self.conn.commit()

    def stats(self):
        rows = self.conn.execute(
            "SELECT product, quantity FROM stock WHERE quantity > 0 ORDER BY quantity DESC, product"
        ).fetchall()
        counts = dict(self.conn.execute("SELECT status, COUNT(*) FROM scans GROUP BY status").fetchall())
        total = sum(counts.values())
        return {
            "products": len(rows),
            "units": sum(q for _, q in rows),
            "stock": rows,
            "scans_total": total,
            "scans": {s: counts.get(s, 0) for s in STATUSES},
            "success_rate": counts.get("confirmed", 0) / total if total else 0.0,
        }

    def recent_scans(self, n=20):
        return self.conn.execute(
            "SELECT ts, source, predicted, confidence, status, final_label, change "
            "FROM scans ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()

    def reset(self):
        self.conn.execute("DELETE FROM stock")
        self.conn.execute("DELETE FROM scans")
        self.conn.commit()
