"""Training loop for the torchvision/timm models (ResNet-50, EfficientNetV2-S).

models/resnet/train.py and models/efficientnet/train.py are both ~10 lines that
call train_torch_model() with a different key. Two copies of this loop would
drift apart and invalidate the comparison.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import hardware
from common.config import set_seed
from common.data import build_dataloaders, class_weights
from common.metrics import epoch_metrics
from common.registry import build_torch_model, split_param_groups
from common.timing import count_parameters, file_size_mb, peak_vram_mb, reset_peak_vram


def _lr_schedule(epoch: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero. Stepped once per epoch."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_torch_model(model_key: str, cfg: dict, epochs: int | None = None) -> dict:
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

    loaders, classes = build_dataloaders(cfg, mcfg["img_size"], batch_size)
    n_classes = len(classes)
    print(f"classes={n_classes}  train={len(loaders['train'].dataset):,}  "
          f"val={len(loaders['val'].dataset):,}  test={len(loaders['test'].dataset):,}")

    model = build_torch_model(mcfg, n_classes, pretrained=True).to(device)
    param_info = count_parameters(model)
    print(f"parameters: {param_info['params_millions']}M")

    weights = class_weights(paths["train"], classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=tcfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        split_param_groups(model, mcfg["source"], mcfg["lr_head"], mcfg["lr_backbone"],
                           tcfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: _lr_schedule(e, tcfg["warmup_epochs"], epochs))

    amp_dt = hardware.amp_dtype()
    use_amp = bool(tcfg["amp"] and amp_dt is not None)
    # fp16 needs loss scaling; bf16 does not. CC 8.9 gives us bf16, so usually off.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dt == torch.float16)
    if use_amp:
        print(f"mixed precision: {str(amp_dt).replace('torch.', '')}")

    ckpt_dir = paths["checkpoints"] / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir = paths["results"] / model_key
    result_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0
    reset_peak_vram()
    train_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- train ----
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dt, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            run_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)
        train_loss, train_acc = run_loss / seen, correct / seen

        # ---- validate ----
        model.eval()
        val_loss, val_seen = 0.0, 0
        y_true, y_pred = [], []
        with torch.inference_mode():
            for images, targets in loaders["val"]:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dt, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_loss += loss.item() * images.size(0)
                val_seen += targets.size(0)
                y_true.append(targets.cpu().numpy())
                y_pred.append(logits.argmax(1).cpu().numpy())
        val_loss /= val_seen
        vm = epoch_metrics(np.concatenate(y_true), np.concatenate(y_pred))

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch + 1, "lr": lr_now,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": vm["accuracy"],
            "val_f1_macro": vm["f1_macro"], "epoch_seconds": round(elapsed, 1),
        })
        flag = ""

        # Early stopping tracks macro-F1, not accuracy: at 191:1 imbalance,
        # accuracy improves while the rare classes get worse.
        if vm["f1_macro"] > best_f1:
            best_f1, best_epoch, epochs_without_gain = vm["f1_macro"], epoch + 1, 0
            torch.save({
                "state_dict": model.state_dict(), "classes": classes,
                "model_key": model_key, "arch": mcfg["arch"], "source": mcfg["source"],
                "img_size": mcfg["img_size"], "epoch": epoch + 1,
                "val_f1_macro": best_f1,
            }, ckpt_dir / "best.pt")
            flag = "  <- best"
        else:
            epochs_without_gain += 1

        print(f"epoch {epoch + 1:3d}/{epochs} | lr {lr_now:.2e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {vm['accuracy']:.4f} macroF1 {vm['f1_macro']:.4f} | "
              f"{elapsed:.0f}s{flag}")

        pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)

        if epochs_without_gain >= tcfg["patience"]:
            print(f"early stopping: no macro-F1 gain for {tcfg['patience']} epochs")
            break

    total_seconds = time.perf_counter() - train_start
    torch.save({
        "state_dict": model.state_dict(), "classes": classes, "model_key": model_key,
        "arch": mcfg["arch"], "source": mcfg["source"], "img_size": mcfg["img_size"],
        "epoch": len(history),
    }, ckpt_dir / "last.pt")

    meta = {
        "model_key": model_key,
        "display_name": mcfg["display_name"],
        "framework": mcfg["source"],
        "arch": mcfg["arch"],
        "img_size": mcfg["img_size"],
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_f1_macro": round(best_f1, 6),
        "training_seconds": round(total_seconds, 1),
        "training_minutes": round(total_seconds / 60, 2),
        "peak_vram_mb": peak_vram_mb(),
        "checkpoint_size_mb": file_size_mb(ckpt_dir / "best.pt"),
        "device": str(device),
        "amp": str(amp_dt).replace("torch.", "") if use_amp else "disabled",
        "class_weighting": "inverse-sqrt frequency, mean-normalized",
        "label_smoothing": tcfg["label_smoothing"],
        "optimizer": "AdamW",
        "lr_head": mcfg["lr_head"],
        "lr_backbone": mcfg["lr_backbone"],
        "schedule": f"{tcfg['warmup_epochs']}-epoch linear warmup then cosine",
        **param_info,
    }
    (result_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (ckpt_dir / "train_config.json").write_text(
        json.dumps({"model": mcfg, "training": tcfg, "seed": cfg["seed"]}, indent=2),
        encoding="utf-8")

    from common.plots import plot_training_curves
    plot_training_curves(pd.DataFrame(history), mcfg["display_name"],
                         paths["plots"] / "training_curves" / f"{model_key}.png")

    print(f"\nDone in {total_seconds / 60:.1f} min. "
          f"Best val macro-F1 {best_f1:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {ckpt_dir / 'best.pt'}")
    return meta
