"""Evaluate the trained YOLO11s-cls on the held-out test set.

Uses the same test file list and the same metric code as every other model
(common/metrics.py), which is what makes the comparison legitimate.

Writes results/yolo11/: metrics.json, classification_report.txt, predictions.csv,
probabilities.npy -- plus the confusion matrix and ROC figures under plots/.

    python models/yolo11/evaluate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import ensure_dirs, load_config  # noqa: E402
from common.evaluate import evaluate_model  # noqa: E402

MODEL_KEY = "yolo11"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if not (cfg["paths"]["checkpoints"] / MODEL_KEY / "best.pt").exists():
        raise SystemExit(f"No checkpoint. Run: python models/yolo11/train.py")

    evaluate_model(MODEL_KEY, cfg)


if __name__ == "__main__":
    main()
