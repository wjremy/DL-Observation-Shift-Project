"""Download the open PhysioNet/CinC Challenge 2012 data (ODC-By 1.0, no account needed) into
data/raw_physionet2012: set-a/b/c record folders and Outcomes-a/b/c.txt. Resumable; the server
sometimes cuts transfers, so each file is retried with HTTP range requests."""
import tarfile
import time
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/challenge-2012/1.0.0/"
OUT = Path(__file__).resolve().parent / "raw_physionet2012"
FILES = ["set-a.tar.gz", "set-b.tar.gz", "set-c.tar.gz", "Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]


def fetch(name: str) -> Path:
    path = OUT / name
    for attempt in range(10):
        done = path.stat().st_size if path.exists() else 0
        request = urllib.request.Request(BASE + name, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, open(path, "ab" if done else "wb") as f:
                if response.status == 416:
                    return path
                while chunk := response.read(1 << 16):
                    f.write(chunk)
            return path
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: attempt {attempt + 1} failed ({exc}); retrying", flush=True)
            time.sleep(3)
    raise RuntimeError(f"Could not download {name}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        p = fetch(name)
        if name.endswith(".tar.gz"):
            with tarfile.open(p) as t:
                t.extractall(OUT)
        print(name, "ok", flush=True)
    for s in "abc":
        print(f"set-{s}:", len(list((OUT / f"set-{s}").glob("*.txt"))), "records (expect 4000)")
