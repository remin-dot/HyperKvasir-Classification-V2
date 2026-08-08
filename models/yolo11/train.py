"""Train YOLO11s-cls on HyperKvasir.

Trained through the Ultralytics API. See common/yolo_trainer.py for the
methodology deviation this implies (Ultralytics owns the optimizer,
schedule and augmentation, and does not support class weighting).

All the real work is in common/ so that all four models share one training loop,
one metric implementation and one data pipeline -- four divergent copies would
make the comparison meaningless.

    python models/yolo11/train.py
    python models/yolo11/train.py --epochs 2      # quick smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import ensure_dirs, load_config  # noqa: E402
from common.train import train_model  # noqa: E402

MODEL_KEY = "yolo11"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, help="override config.yaml epochs")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if not cfg["paths"]["train"].exists():
        raise SystemExit("Dataset not prepared. Run: python scripts/prepare_dataset.py")

    train_model(MODEL_KEY, cfg, args.epochs)


if __name__ == "__main__":
    main()
