"""Generate Grad-CAM visualizations for every evaluated model.

    python scripts/generate_gradcam.py
    python scripts/generate_gradcam.py --only resnet --correct 5 --incorrect 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import hardware  # noqa: E402
from common.cam import run_gradcam  # noqa: E402
from common.config import ensure_dirs, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="MODEL")
    ap.add_argument("--correct", type=int, default=3, help="correctly classified examples")
    ap.add_argument("--incorrect", type=int, default=3, help="misclassified examples")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    device = hardware.setup_torch(cfg["seed"])
    keys = args.only or list(cfg["models"].keys())

    summaries, skipped = [], []
    for key in keys:
        if not (cfg["paths"]["results"] / key / "predictions.csv").exists():
            print(f"[skip] {key}: not evaluated yet")
            skipped.append(key)
            continue
        try:
            summaries.append(run_gradcam(key, cfg, args.correct, args.incorrect, device))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            (cfg["paths"]["logs"] / f"{key}_gradcam_error.log").write_text(tb, encoding="utf-8")
            record = {"model": key, "method": None, "images": 0,
                      "error_type": type(exc).__name__, "error": str(exc),
                      "note": "Explainability could not be produced for this model; "
                              "see the log for the traceback."}
            (cfg["paths"]["gradcam"] / key / "gradcam_info.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8")
            summaries.append(record)
            print(f"GRAD-CAM FAILED for {key}: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 70}\nGRAD-CAM SUMMARY\n{'=' * 70}")
    for s in summaries:
        print(f"{s['model']:<16} method={s.get('method') or 'FAILED':<10} images={s['images']}")
    if skipped:
        print(f"skipped (not evaluated): {skipped}")
    print("\nNext: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
