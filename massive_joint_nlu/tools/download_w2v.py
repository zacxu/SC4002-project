"""Download NLPL Word2Vec model 6 (English Wikipedia 2017, 300d) with resume support."""
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://vectors.nlpl.eu/repository/20/6.zip"
dest_dir = Path(sys.argv[1])
dest_dir.mkdir(parents=True, exist_ok=True)
zip_path = dest_dir / "6.zip"

head = urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=30)
total = int(head.headers.get("Content-Length", "0"))
print("remote size", total, flush=True)

have = zip_path.stat().st_size if zip_path.exists() else 0
while have < total:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(zip_path, "ab") as f:
            t0 = time.time()
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                if int(time.time() - t0) % 10 == 0:
                    print(f"{have/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    except Exception as e:  # resume on transient failure
        print("retry after error:", e, flush=True)
        time.sleep(5)
        have = zip_path.stat().st_size if zip_path.exists() else 0

print("download complete", have, flush=True)
with zipfile.ZipFile(zip_path) as zf:
    print(zf.namelist())
    zf.extractall(dest_dir)
for p in dest_dir.iterdir():
    print(p.name, p.stat().st_size)
print("DONE")
