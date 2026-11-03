"""Resumable ranged download of the official CIFAR-10 archive with MD5 validation."""
import concurrent.futures
import hashlib
import math
from pathlib import Path

import requests

URL = "https://hf-mirror.com/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"
OUT = Path(__file__).parent / "data" / "cifar-10-python.tar.gz"
PARTS = OUT.parent / "parts"
CHUNK = 2 * 1024 * 1024


def get_part(item):
    i, size = item
    start = i * CHUNK
    end = min(size - 1, start + CHUNK - 1)
    target = PARTS / f"{i:04d}.part"
    expected = end - start + 1
    if target.exists() and target.stat().st_size == expected:
        return i
    for attempt in range(8):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 180))
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != expected:
                raise RuntimeError(f"part {i}: status {r.status_code}, length {len(r.content)}")
            target.write_bytes(r.content)
            return i
        except Exception as e:
            print(f"retry {i} {attempt+1}: {e}", flush=True)
    raise RuntimeError(f"part {i} failed")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        existing = hashlib.md5(OUT.read_bytes()).hexdigest()
        if existing == MD5:
            print(f"OK {OUT} MD5={MD5} (already present)", flush=True)
            return
    PARTS.mkdir(exist_ok=True)
    head = requests.get(URL, headers={"Range": "bytes=0-0"}, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Range"].split("/")[-1])
    jobs = [(i, size) for i in range(math.ceil(size / CHUNK))]
    print(f"size={size}, parts={len(jobs)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, i in enumerate(pool.map(get_part, jobs), 1):
            print(f"{n}/{len(jobs)} part={i}", flush=True)
    md5 = hashlib.md5()
    with OUT.open("wb") as f:
        for i, _ in jobs:
            data = (PARTS / f"{i:04d}.part").read_bytes()
            f.write(data)
            md5.update(data)
    if md5.hexdigest() != MD5:
        OUT.unlink()
        raise RuntimeError(f"MD5 mismatch: {md5.hexdigest()}")
    print(f"OK {OUT} MD5={MD5}", flush=True)


if __name__ == "__main__":
    main()
