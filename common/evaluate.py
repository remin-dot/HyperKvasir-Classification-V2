"""Unified test-set evaluation.

Every model is scored here, on the same ordered file list, with the same metric
code. Nothing model-specific happens beyond calling the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import hardware
from common.config import class_names
from common.data import test_image_paths
from common.metrics import compute_all, summary_line
from common.plots import plot_confusion_matrix, plot_roc_curves
from common.registry import load_model
from common.timing import file_size_mb, measure_latency, peak_vram_mb, reset_peak_vram, timer


def evaluate_model(model_key: str, cfg: dict, device=None) -> dict:
    device = device or hardware.setup_torch(cfg["seed"])
    classes = class_names(cfg)
    mcfg = cfg["models"][model_key]

    print(f"\n{'=' * 70}\nEvaluating {mcfg['display_name']} ({model_key})\n{'=' * 70}")

    adapter = load_model(model_key, cfg, classes, device)
    paths, y_true = test_image_paths(cfg, classes)
    print(f"test images: {len(paths):,}  classes: {len(classes)}")

    batch = hardware.pick_batch_size(mcfg["batch_size"], hardware.usable_vram_gb(), device.type == "cuda")
    reset_peak_vram()
    with timer("inference over test set") as t:
        y_prob = adapter.predict_proba(paths, batch_size=batch)
    eval_vram = peak_vram_mb()

    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError(f"{model_key}: probability rows do not sum to 1 -- adapter bug")

    metrics = compute_all(y_true, y_prob, classes)

    # Latency: identical protocol for all four models -- raw forward on one
    # pre-processed tensor, no disk I/O, no framework post-processing.
    sample = adapter.sample_tensor(paths[0])
    latency = measure_latency(
        adapter.forward_tensor, sample,
        warmup=cfg["evaluation"]["latency_warmup"],
        iters=cfg["evaluation"]["latency_iters"],
    )

    train_meta = {}
    meta_path = cfg["paths"]["results"] / model_key / "train_meta.json"
    if meta_path.exists():
        train_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    n_params = adapter.n_params()
    metrics.update({
        "model_key": model_key,
        "display_name": mcfg["display_name"],
        "framework": mcfg["source"],
        "img_size": mcfg["img_size"],
        "params_total": int(n_params),
        "params_millions": round(n_params / 1e6, 2),
        "checkpoint_size_mb": file_size_mb(adapter.ckpt_path),
        "checkpoint_path": str(adapter.ckpt_path),
        "training_seconds": train_meta.get("training_seconds"),
        "training_minutes": train_meta.get("training_minutes"),
        "train_peak_vram_mb": train_meta.get("peak_vram_mb"),
        "eval_peak_vram_mb": eval_vram,
        "epochs_run": train_meta.get("epochs_run"),
        "device": str(device),
        "test_inference_seconds": round(t["seconds"], 2),
        **latency,
    })

    result_dir = cfg["paths"]["results"] / model_key
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / "classification_report.txt").write_text(
        f"{mcfg['display_name']} — HyperKvasir test set ({len(paths)} images)\n\n"
        + metrics["classification_report"], encoding="utf-8")

    # Per-image predictions: consumed by gradcam.py to find correct/incorrect examples.
    y_pred = y_prob.argmax(axis=1)
    pd.DataFrame({
        "path": paths,
        "true_idx": y_true,
        "true_class": [classes[i] for i in y_true],
        "pred_idx": y_pred,
        "pred_class": [classes[i] for i in y_pred],
        "confidence": y_prob.max(axis=1),
        "correct": y_true == y_pred,
    }).to_csv(result_dir / "predictions.csv", index=False)
    np.save(result_dir / "probabilities.npy", y_prob)

    plots = cfg["paths"]["plots"]
    plot_confusion_matrix(metrics["confusion_matrix"], classes, mcfg["display_name"],
                          plots / "confusion_matrices" / f"{model_key}.png", normalize=True)
    plot_confusion_matrix(metrics["confusion_matrix"], classes, mcfg["display_name"],
                          plots / "confusion_matrices" / f"{model_key}_counts.png", normalize=False)
    plot_roc_curves(y_true, y_prob, classes, mcfg["display_name"],
                    plots / "roc_curves" / f"{model_key}.png")

    print("\n" + summary_line(mcfg["display_name"], metrics))
    print(f"latency {latency['inference_ms_mean']:.2f} ms/img ({latency['fps']} FPS)  "
          f"params {metrics['params_millions']}M  ckpt {metrics['checkpoint_size_mb']} MB")
    print(f"-> {result_dir / 'metrics.json'}")

    # Rare classes have 1-2 test images; their per-class numbers are noise, and
    # saying so here means nobody has to discover it from the report alone.
    tiny = [c for c, v in metrics["per_class"].items() if v["support"] <= 2]
    if tiny:
        print(f"\nNOTE: {len(tiny)} class(es) have <=2 test images "
              f"({', '.join(tiny)}); their per-class scores are not statistically meaningful.")
    return metrics
