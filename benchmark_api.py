import argparse
import statistics
import time

import requests


def main():
    parser = argparse.ArgumentParser(description="Measure /predict latency over repeated requests")
    parser.add_argument("image")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()

    requests.post(args.url, files={"file": ("img.jpg", data)}).raise_for_status()

    times = []
    for _ in range(args.n):
        start = time.perf_counter()
        requests.post(args.url, files={"file": ("img.jpg", data)}).raise_for_status()
        times.append((time.perf_counter() - start) * 1000)

    print(f"requests: {args.n}")
    print(f"average latency: {statistics.mean(times):.1f} ms")
    print(f"median latency:  {statistics.median(times):.1f} ms")


if __name__ == "__main__":
    main()
