"""Training wrapper for the Ultralytics classification models (YOLOv8-cls, YOLO11-cls).

IMPORTANT METHODOLOGICAL NOTE, reproduced in FINAL_REPORT.md section 3:

Ultralytics owns its own training loop. It will not accept our optimizer, our LR
schedule, our augmentation policy or our class weights. So the YOLO models are
NOT trained under an identical recipe to ResNet-50 and EfficientNetV2-S, and
claiming otherwise would be false.

What IS identical, and what the comparison actually rests on:
  * the same train/val/test split, from the same files on disk
  * the same number of epochs and the same early-stopping patience
  * the same held-out test set, scored by the same code in common/metrics.py

Ultralytics also reports only top-1/top-5 accuracy per epoch, not macro-F1, so
its training curves show accuracy where the torch models show macro-F1.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

from common import hardware
from common.config import class_names, set_seed
from common.timing import file_size_mb, peak_vram_mb, reset_peak_vram


def _normalize_history(results_csv: Path) -> pd.DataFrame:
    """Convert Ultralytics' results.csv into our common history schema.

    Column names have carried leading spaces and have been renamed between
    releases, so match loosely instead of indexing fixed names.
    """
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    def pick(*candidates):
        for cand in candidates:
            for col in df.columns:
                if col.lower() == cand.lower():
                    return df[col]
        return None

    out = pd.DataFrame()
    out["epoch"] = pick("epoch") if pick("epoch") is not None else range(1, len(df) + 1)
    for name, cands in {
        "train_loss": ("train/loss",),
        "val_loss": ("val/loss",),
        "val_acc": ("metrics/accuracy_top1", "metrics/accuracy_top1(B)"),
        "val_acc_top5": ("metrics/accuracy_top5",),
        "lr": ("lr/pg0",),
    }.items():
        series = pick(*cands)
        if series is not None:
            out[name] = series
    return out


def train_yolo_model(model_key: str, cfg: dict, epochs: int | None = None) -> dict:
    from ultralytics import YOLO

    tcfg = cfg["training"]
    mcfg = cfg["models"][model_key]
    epochs = epochs or tcfg["epochs"]
    paths = cfg["paths"]

    set_seed(cfg["seed"])
    device = hardware.setup_torch(cfg["seed"])
    hw = hardware.detect(paths["data_root"])
    batch_size = hardware.pick_batch_size(
        mcfg["batch_size"], hardware.usable_vram_gb(hw), device.type == "cuda")

    print(f"\n{'=' * 70}\nTraining {mcfg['display_name']} ({model_key})\n{'=' * 70}")
    print(f"device={device}  img_size={mcfg['img_size']}  batch={batch_size}  epochs={epochs}")
    print("NOTE: Ultralytics uses its own optimizer/augmentation policy -- "
          "see the methodology note in this file and in FINAL_REPORT.md.")

    classes = class_names(cfg)
    run_dir = paths["logs"] / "ultralytics"
    ckpt_dir = paths["checkpoints"] / model_key
    result_dir = paths["results"] / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(mcfg["weights"])           # downloads pretrained weights on first use
    reset_peak_vram()
    start = time.perf_counter()

    model.train(
        data=str(paths["data_root"]),        # dir containing train/ val/ test/
        epochs=epochs,
        imgsz=mcfg["img_size"],
        batch=batch_size,
        patience=tcfg["patience"],
        device=0 if device.type == "cuda" else "cpu",
        workers=min(tcfg["num_workers"], 8),
        seed=cfg["seed"],
        amp=bool(tcfg["amp"]),
        project=str(run_dir),
        name=model_key,
        exist_ok=True,
        verbose=True,
        plots=False,                          # we draw our own, consistently
    )
    total_seconds = time.perf_counter() - start

    save_dir = Path(model.trainer.save_dir)
    for src_name, dst_name in (("best.pt", "best.pt"), ("last.pt", "last.pt")):
        src = save_dir / "weights" / src_name
        if src.exists():
            shutil.copy2(src, ckpt_dir / dst_name)
        else:
            raise FileNotFoundError(f"Ultralytics did not produce {src}")

    history = pd.DataFrame()
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        history = _normalize_history(results_csv)
        history.to_csv(result_dir / "training_history.csv", index=False)
        from common.plots import plot_training_curves
        plot_training_curves(history, mcfg["display_name"],
                             paths["plots"] / "training_curves" / f"{model_key}.png")

    import torch
    n_params = sum(p.numel() for p in model.model.parameters())
    best_acc = float(history["val_acc"].max()) if "val_acc" in history else None

    meta = {
        "model_key": model_key,
        "display_name": mcfg["display_name"],
        "framework": "ultralytics",
        "arch": mcfg["weights"],
        "img_size": mcfg["img_size"],
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": int(len(history)) if len(history) else epochs,
        "best_val_top1_acc": best_acc,
        "training_seconds": round(total_seconds, 1),
        "training_minutes": round(total_seconds / 60, 2),
        "peak_vram_mb": peak_vram_mb(),
        "checkpoint_size_mb": file_size_mb(ckpt_dir / "best.pt"),
        "device": str(device),
        "amp": "ultralytics-managed" if tcfg["amp"] else "disabled",
        "params_total": int(n_params),
        "params_millions": round(n_params / 1e6, 2),
        "ultralytics_run_dir": str(save_dir),
        "methodology_deviation": (
            "Optimizer, LR schedule and augmentation are Ultralytics defaults and could "
            "not be matched to the torch models. Class weighting is not supported by the "
            "Ultralytics classification trainer, so this model trains unweighted. Split, "
            "epoch budget, patience and test-set evaluation are identical."
        ),
        "n_classes": len(classes),
    }
    (result_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {total_seconds / 60:.1f} min. Checkpoint: {ckpt_dir / 'best.pt'}")
    return meta
