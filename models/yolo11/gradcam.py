"""Grad-CAM explainability for YOLO11s-cls.

Produces original / heatmap / overlay triptychs for correctly classified and
misclassified test images, into gradcam/yolo11/.

The misclassified examples are the useful half: they show whether the model is
looking at mucosa or at the black frame border / green ScopeGuide overlay.

    python models/yolo11/gradcam.py
    python models/yolo11/gradcam.py --correct 5 --incorrect 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.cam import run_gradcam  # noqa: E402
from common.config import ensure_dirs, load_config  # noqa: E402

MODEL_KEY = "yolo11"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--correct", type=int, default=3)
    ap.add_argument("--incorrect", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if not (cfg["paths"]["results"] / MODEL_KEY / "predictions.csv").exists():
        raise SystemExit(f"Not evaluated yet. Run: python models/yolo11/evaluate.py")

    run_gradcam(MODEL_KEY, cfg, args.correct, args.incorrect)


if __name__ == "__main__":
    main()
