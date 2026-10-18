"""Download the PhysioNet/CinC Challenge 2019 sepsis training data (open access, CC BY 4.0).

PhysioNet offers no ZIP for this project, so the 40,336 patient files (.psv) are
fetched one by one from https://physionet.org/files/challenge-2019/1.0.0/training/
into data/raw_physionet2019/training_setA/ and training_setB/. Resumable: files
that already exist are skipped. Usage: python data/fetch_physionet2019.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import urllib.request

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
ROOT = Path(__file__).resolve().parent / "raw_physionet2019"
WORKERS = 12


def listing(subset):
    cached = ROOT / f"list_{subset[-1]}.txt"
    if cached.exists():
        return cached.read_text().split()
    html = urllib.request.urlopen(BASE + subset + "/", timeout=300).read().decode()
    names = sorted(set(re.findall(r'href="(p\d+\.psv)"', html)))
    cached.write_text("\n".join(names))
    return names


def fetch(job):
    subset, name = job
    target = ROOT / subset / name
    if target.exists() and target.stat().st_size > 0:
        return 0
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(BASE + f"{subset}/{name}", timeout=60).read()
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            return 1
        except Exception:
            if attempt == 2:
                print(f"FAILED {subset}/{name}", flush=True)
    return 0


def main():
    jobs = []
    for subset in ("training_setA", "training_setB"):
        (ROOT / subset).mkdir(parents=True, exist_ok=True)
        jobs += [(subset, name) for name in listing(subset)]
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, got in enumerate(pool.map(fetch, jobs), 1):
            done += got
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} checked, {done} downloaded", flush=True)
    missing = [j for j in jobs if not (ROOT / j[0] / j[1]).exists()]
    print(f"finished: {len(jobs) - len(missing)}/{len(jobs)} present, {len(missing)} missing", flush=True)


if __name__ == "__main__":
    main()
