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


def _curl_download(url: str, dest: Path) -> bool:
    """Try curl first. Returns False if curl is unavailable or fails.

    datasets.simula.no serves an incomplete TLS chain (it omits the intermediate
    certificate). Browsers and curl recover by fetching the missing intermediate
    via the certificate's AIA extension; Python's ssl module does not, and fails
    with CERTIFICATE_VERIFY_FAILED. curl ships with Windows 10/11 and with most
    Linux/macOS installs, and still performs full certificate validation -- so
    this is a fix, not a bypass.
    """
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if not curl:
        return False

    print("Downloading via curl (resumable, shows progress) ...")
    result = subprocess.run([
        curl, "-L", "--fail", "--retry", "3", "--retry-delay", "5",
        "-C", "-",                     # resume if a partial file exists
        "--connect-timeout", "30",
        "-o", str(dest), url,
    ])
    if result.returncode == 0:
        return True
    # curl exits 33 when the server refuses a range request and 416 when the
    # file is already complete; both are recoverable, everything else is not.
    if result.returncode in (33, 36) and dest.exists():
        print("Resume rejected by server; restarting the download.")
        dest.unlink()
        return subprocess.run([curl, "-L", "--fail", "-o", str(dest), url]).returncode == 0
    print(f"curl failed (exit {result.returncode}); falling back to urllib.")
    return False


def download(url: str, dest: Path) -> Path:
    """Stream to disk with an HTTP Range resume."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _curl_download(url, dest):
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
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"Downloading {args.component} ({spec['size_gb']} GB) -> {zip_path}")
        download(spec["url"], zip_path)
    else:
        print(f"Using existing archive: {zip_path} ({_human(zip_path.stat().st_size)})")

    if args.no_extract:
        print("--no-extract: stopping before extraction.")
        return

    extract(zip_path, target)
    n = sum(1 for _ in target.rglob("*.jpg"))
    print(f"\n{n:,} JPEG images now under {target}")


if __name__ == "__main__":
    main()
