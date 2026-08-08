"""Evaluate every trained model on the identical held-out test set.

    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --only resnet
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import hardware  # noqa: E402
from common.config import ensure_dirs, load_config  # noqa: E402
from common.evaluate import evaluate_model  # noqa: E402
from common.metrics import summary_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="MODEL")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if metrics.json exists")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    device = hardware.setup_torch(cfg["seed"])
    keys = args.only or list(cfg["models"].keys())

    done, failed, skipped = {}, [], []
    for key in keys:
        metrics_path = cfg["paths"]["results"] / key / "metrics.json"
        if metrics_path.exists() and not args.force:
            print(f"[skip] {key}: {metrics_path} exists (--force to redo)")
            done[key] = json.loads(metrics_path.read_text(encoding="utf-8"))
            skipped.append(key)
            continue
        if not (cfg["paths"]["checkpoints"] / key / "best.pt").exists():
            print(f"[skip] {key}: no checkpoint -- not trained (or training failed)")
            failed.append(key)
            continue
        try:
            done[key] = evaluate_model(key, cfg, device)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            (cfg["paths"]["results"] / key / "FAILED.json").write_text(json.dumps({
                "model_key": key, "stage": "evaluation",
                "error_type": type(exc).__name__, "error": str(exc), "traceback": tb,
            }, indent=2), encoding="utf-8")
            (cfg["paths"]["logs"] / f"{key}_eval_error.log").write_text(tb, encoding="utf-8")
            print(f"\nEVALUATION FAILED: {key} — {type(exc).__name__}: {exc}\n")
            failed.append(key)

    print(f"\n{'=' * 70}\nEVALUATION SUMMARY  (test set, identical for all models)\n{'=' * 70}")
    for key, m in sorted(done.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True):
        print(summary_line(m["display_name"], m))
    if failed:
        print(f"\nnot evaluated: {failed}")
    print("\nNext: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
