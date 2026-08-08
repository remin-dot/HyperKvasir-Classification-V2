"""Download and extract the HyperKvasir labeled-image archive (~4 GB, CC BY 4.0).

Resumable: re-running after an interrupted download continues where it stopped
rather than starting over.

    python scripts/download_dataset.py
    python scripts/download_dataset.py --zip "C:/path/to/already-downloaded.zip"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# Abort and reconnect if throughput stays under this for MIN_SPEED_WINDOW seconds.
# See _curl_download for why.
MIN_SPEED_BYTES = 50_000
MIN_SPEED_WINDOW = 30


def _curl_download(url: str, dest: Path, expected_size: int | None = None) -> bool:
    """Try curl first. Returns False if curl is unavailable or fails.

    Two properties of datasets.simula.no are worked around here.

    1. TLS. The host serves an incomplete certificate chain (it omits the
       intermediate). Browsers and curl recover by fetching the missing
       intermediate via the certificate's AIA extension; Python's ssl module
       does not, and fails with CERTIFICATE_VERIFY_FAILED. curl ships with
       Windows 10/11 and most Linux/macOS installs and still performs full
       certificate validation -- a fix, not a bypass.

    2. Throughput decay on long-lived connections. Measured on a 29 GB
       transfer: a fresh connection sustains ~300 KB/s, the same connection an
       hour later delivers ~2 KB/s -- an ETA of 133 days. Reconnecting restores
       full speed immediately. So rather than one long transfer, this runs a
       loop of resumed transfers and deliberately aborts any connection whose
       throughput collapses (--speed-limit / --speed-time). Each restart is a
       fresh connection continuing from the current byte offset via -C -.

    Without (2) the large archives simply do not finish.
    """
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        return False

    print("Downloading via curl (resumable; reconnects when throughput decays) ...")
    attempt = 0
    while True:
        attempt += 1
        before = dest.stat().st_size if dest.exists() else 0
        if expected_size and before >= expected_size:
            return True

        result = subprocess.run([
            curl, "-L", "--fail",
            "-C", "-",                          # resume from the current offset
            "--retry", "5", "--retry-delay", "10", "--retry-all-errors",
            "--connect-timeout", "30",
            "--speed-limit", str(MIN_SPEED_BYTES),
            "--speed-time", str(MIN_SPEED_WINDOW),
            "-o", str(dest), url,
        ])
        after = dest.stat().st_size if dest.exists() else 0

        if result.returncode == 0:
            return True
        # 28 == operation timed out, which is what --speed-limit raises. That is
        # the expected path here, not an error: reconnect and keep going.
        if result.returncode == 28:
            gained = (after - before) / 1024**2
            print(f"  [reconnect {attempt}] throughput decayed; "
                  f"+{gained:.0f} MB this leg, {after / 1024**3:.2f} GB total")
            if gained <= 0 and attempt > 20:
                print("  no progress across many reconnects; giving up on curl.")
                return False
            continue
        # 33/36: server refused the range request. Restart from zero.
        if result.returncode in (33, 36) and dest.exists():
            print("Resume rejected by server; restarting the download.")
            dest.unlink()
            continue
        print(f"curl failed (exit {result.returncode}); falling back to urllib.")
        return False


def download(url: str, dest: Path, expected_size: int | None = None) -> Path:
    """Stream to disk with an HTTP Range resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _curl_download(url, dest, expected_size):
        return dest

    existing = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "hyperkvasir-benchmark/1.0"})
    if existing:
        print(f"Resuming from {_human(existing)} ...")
        req.add_header("Range", f"bytes={existing}-")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            if existing and response.status != 206:
                print("Server ignored the resume request; restarting the download.")
                existing = 0
            total = int(response.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing and response.status == 206 else "wb"
            done = existing if mode == "ab" else 0

            with open(dest, mode) as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    pct = f"{done / total * 100:5.1f}%" if total else "  ?  "
                    print(f"\r  {pct}  {_human(done)}" + (f" / {_human(total)}" if total else ""),
                          end="", flush=True)
            print()
        return dest

    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        print(f"\nDownload failed: {reason}\n")
        print("=" * 70)
        print("MANUAL DOWNLOAD")
        print("=" * 70)
        print("This host has been known to present an incomplete TLS certificate chain")
        print("on some Windows setups. A browser handles it fine. To continue:")
        print(f"  1. Open  {url}")
        print(f"  2. Save the zip to  {dest}")
        print("  3. Re-run this script (it will skip the download and just extract),")
        print(f"     or point at it directly:  python scripts/download_dataset.py --zip \"{dest}\"")
        print("=" * 70)
        raise SystemExit(1)


def extract(zip_path: Path, raw_dir: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        raise SystemExit(f"Not a valid zip archive (download likely truncated): {zip_path}\n"
                         f"Delete it and re-run to download again.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        print(f"Extracting {len(members):,} entries to {raw_dir} ...")
        zf.extractall(raw_dir)
    print("Extraction complete.")


# Where each component of the 58.6 GB release extracts to, and the zip filename
# it arrives as. Downloaded piecemeal rather than as the single hyper-kvasir.zip:
# one failed resume on a 58 GB file over a 300 KB/s link costs more than a day.
COMPONENT_TARGETS = {
    "labeled":   ("hyper-kvasir-labeled-images.zip",   "raw"),
    "unlabeled": ("hyper-kvasir-unlabeled-images.zip", "raw_unlabeled"),
    "segmented": ("hyper-kvasir-segmented-images.zip", "raw_segmented"),
    "videos":    ("hyper-kvasir-videos.zip",           "raw_videos"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--component", default="labeled", choices=sorted(COMPONENT_TARGETS),
                    help="which archive of the HyperKvasir release to fetch")
    ap.add_argument("--zip", type=Path, help="use an already-downloaded archive")
    ap.add_argument("--no-extract", action="store_true",
                    help="download only; extract later (useful while a 29 GB "
                         "transfer is still running)")
    ap.add_argument("--force", action="store_true", help="re-extract even if images exist")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    zip_name, target_key = COMPONENT_TARGETS[args.component]
    target = cfg["paths"][target_key]
    spec = cfg["dataset"]["components"][args.component]

    existing = sum(1 for _ in target.rglob("*.jpg")) if target.exists() else 0
    if existing > 1000 and not args.force:
        print(f"{existing:,} images already present in {target}. Nothing to do (--force to redo).")
        return

    need_gb = spec["size_gb"] * 2.1          # archive + extracted copy
    free_gb = shutil.disk_usage(cfg["paths"]["data_root"]).free / 1024**3
    if free_gb < need_gb:
        print(f"WARNING: {free_gb:.1f} GB free, this component needs roughly "
              f"{need_gb:.1f} GB (archive + extracted).")

    zip_path = args.zip or (cfg["paths"]["archives"] / zip_name)
    # Resume on partial files. Testing only for existence would treat a 164 MB
    # fragment of a 29 GB archive as finished and hand a truncated zip to
    # extract(). size_gb is approximate, so allow 5% either way.
    approx = spec["size_gb"] * 1024**3
    have = zip_path.stat().st_size if zip_path.exists() else 0
    if have < approx * 0.95:
        if have:
            print(f"Resuming {args.component}: {_human(have)} of ~{spec['size_gb']} GB already on disk")
        else:
            print(f"Downloading {args.component} ({spec['size_gb']} GB) -> {zip_path}")
        download(spec["url"], zip_path, int(approx))
    else:
        print(f"Using existing archive: {zip_path} ({_human(have)})")

    if args.no_extract:
        print("--no-extract: stopping before extraction.")
        return

    extract(zip_path, target)
    n = sum(1 for _ in target.rglob("*.jpg"))
    print(f"\n{n:,} JPEG images now under {target}")


if __name__ == "__main__":
    main()
