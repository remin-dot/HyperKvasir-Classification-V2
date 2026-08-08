"""Train every model in config.yaml, in order.

A model that fails does not stop the others: the traceback is recorded and the
final report says which model failed and why, rather than omitting it silently.

    python scripts/train_all.py
    python scripts/train_all.py --only resnet efficientnet --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402
from common.train import train_model  # noqa: E402


def train_one(model_key: str, cfg: dict, epochs: int | None) -> tuple[bool, dict]:
    """Returns (succeeded, meta-or-failure-record)."""
    result_dir = cfg["paths"]["results"] / model_key
    result_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = result_dir / "FAILED.json"
    if failed_marker.exists():
        failed_marker.unlink()

    try:
        return True, train_model(model_key, cfg, epochs)
    except Exception as exc:  # noqa: BLE001 - any failure must be recorded, not raised
        tb = traceback.format_exc()
        record = {
            "model_key": model_key,
            "display_name": cfg["models"][model_key]["display_name"],
            "stage": "training",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": tb,
        }
        failed_marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
        log = cfg["paths"]["logs"] / f"{model_key}_train_error.log"
        log.write_text(tb, encoding="utf-8")
        print(f"\n{'!' * 70}\nTRAINING FAILED: {model_key}\n{type(exc).__name__}: {exc}")
        print(f"Full traceback: {log}\nContinuing with the remaining models.\n{'!' * 70}\n")
        return False, record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="MODEL", help="subset of model keys")
    ap.add_argument("--epochs", type=int, help="override config.yaml epochs")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    keys = args.only or list(cfg["models"].keys())
    unknown = [k for k in keys if k not in cfg["models"]]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Known: {list(cfg['models'])}")

    if not cfg["paths"]["train"].exists():
        raise SystemExit("Dataset not prepared. Run: python scripts/prepare_dataset.py")

    succeeded, failed, skipped = [], [], []
    for key in keys:
        ckpt = cfg["paths"]["checkpoints"] / key / "best.pt"
        if ckpt.exists() and not args.force:
            print(f"\n[skip] {key}: checkpoint already at {ckpt} (--force to retrain)")
            skipped.append(key)
            continue
        ok, _ = train_one(key, cfg, args.epochs)
        (succeeded if ok else failed).append(key)

    print(f"\n{'=' * 70}\nTRAINING SUMMARY\n{'=' * 70}")
    print(f"trained : {succeeded or '-'}")
    print(f"skipped : {skipped or '-'}")
    print(f"failed  : {failed or '-'}")
    if failed:
        print("\nFailures are recorded in results/<model>/FAILED.json and appear in "
              "FINAL_REPORT.md. No fake metrics are produced for them.")


if __name__ == "__main__":
    main()
