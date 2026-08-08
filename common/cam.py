"""Grad-CAM explainability for all four models.

ResNet-50 and EfficientNetV2-S are plain nn.Modules, so Grad-CAM applies directly.

Ultralytics ships no CAM support. The adapter reaches into YOLO(...).model, which
IS a plain nn.Module returning class scores, and hooks the last convolutional
block before the Classify head (the head itself pools away all spatial extent, so
a heatmap taken there would be a single pixel). If gradient hooks fail on the
Ultralytics graph -- it has changed shape between releases -- we fall back to
EigenCAM, which needs activations only and no gradients, and record the
substitution so the report states which method produced each figure.

These figures are also the check on the green ScopeGuide overlay: if attention
sits on a corner box or the image border instead of mucosa, the metrics are
measuring the wrong thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from common import hardware
from common.config import class_names
from common.registry import load_model


class _CamWrapper(nn.Module):
    """Normalizes model output to a plain [N, C] tensor of logit-scale scores.

    Two problems this solves:

    1. Ultralytics classification forward returns a bare tensor in some releases
       and a (tensor, ...) tuple in others; pytorch-grad-cam requires the tensor.

    2. Ultralytics' Classify head applies softmax internally, so its output is a
       probability distribution, not logits. Grad-CAM differentiates the target
       class score with respect to the layer activations -- and the gradient of a
       saturated softmax probability underflows to zero, producing an all-zero
       heatmap that looks like a broken hook rather than a math problem.
       Taking the log recovers logit-scale values (log-softmax differs from the
       logits only by an additive constant, which gradients ignore) and restores
       usable gradients. Verified: without this, GradCAM on yolov8s-cls returns a
       heatmap with range [0.000, 0.000].

    Models that already emit logits (ResNet-50, EfficientNetV2-S) are passed
    through untouched.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        while isinstance(out, (list, tuple)):
            out = out[0]
        if out.dim() == 2 and bool(torch.all(out >= 0)):
            sums = out.sum(dim=1)
            if bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3)):
                return torch.log(out.clamp_min(1e-12))
        return out


def _display_image(path: str, size: int) -> np.ndarray:
    """Reproduce the eval transform's geometry (resize short side, center crop)
    so the heatmap lines up with what the model actually saw."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        target = int(size * 1.14)
        scale = target / min(w, h)
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        return np.asarray(im, dtype=np.float32) / 255.0


def _pick_examples(preds: pd.DataFrame, n: int, correct: bool) -> pd.DataFrame:
    """Most confident examples, preferring distinct classes for variety.

    For mistakes, high confidence is the point: a confidently wrong prediction is
    the diagnostically interesting failure.
    """
    subset = preds[preds["correct"] == correct].sort_values("confidence", ascending=False)
    if subset.empty:
        return subset
    picked = subset.drop_duplicates(subset="true_class", keep="first").head(n)
    if len(picked) < n:                       # not enough distinct classes; top up
        rest = subset.drop(picked.index).head(n - len(picked))
        picked = pd.concat([picked, rest])
    return picked


def _triptych(rgb: np.ndarray, heat: np.ndarray, overlay: np.ndarray,
              row: pd.Series, method: str, model_name: str, out: Path) -> None:
    ok = bool(row["correct"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, img, title in zip(axes, [rgb, heat, overlay],
                              ["original", f"{method} heatmap", "overlay"]):
        ax.imshow(img, cmap="jet" if title.endswith("heatmap") else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    verdict = "CORRECT" if ok else "MISCLASSIFIED"
    fig.suptitle(
        f"{model_name} — {verdict}\n"
        f"true: {row['true_class']}   |   predicted: {row['pred_class']}   "
        f"|   confidence: {row['confidence']:.3f}",
        fontsize=11, color="darkgreen" if ok else "darkred")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_gradcam(model_key: str, cfg: dict, n_correct: int = 3, n_incorrect: int = 3,
                device=None) -> dict:
    from pytorch_grad_cam import EigenCAM, GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    device = device or hardware.setup_torch(cfg["seed"])
    classes = class_names(cfg)
    mcfg = cfg["models"][model_key]
    out_dir = cfg["paths"]["gradcam"] / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}\nGrad-CAM: {mcfg['display_name']} ({model_key})\n{'=' * 70}")

    preds_csv = cfg["paths"]["results"] / model_key / "predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(
            f"{preds_csv} missing. Run: python scripts/evaluate_all.py --only {model_key}")
    preds = pd.read_csv(preds_csv)

    adapter = load_model(model_key, cfg, classes, device)
    model = _CamWrapper(adapter.cam_model()).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)                # Ultralytics may ship weights with grads off
    target_layers = adapter.cam_target_layers()
    size = mcfg["img_size"]

    method_name, cam_ctor = "Grad-CAM", GradCAM
    examples = pd.concat([
        _pick_examples(preds, n_correct, correct=True),
        _pick_examples(preds, n_incorrect, correct=False),
    ])
    if examples.empty:
        print("no test predictions to visualize")
        return {"model": model_key, "method": None, "images": 0}

    written, fallback_used = [], False
    for _, row in examples.iterrows():
        tensor = adapter.sample_tensor(row["path"])
        rgb = _display_image(row["path"], size)
        target = [ClassifierOutputTarget(int(row["pred_idx"]))]

        try:
            with cam_ctor(model=model, target_layers=target_layers) as cam:
                grayscale = cam(input_tensor=tensor, targets=target)[0]
        except Exception as exc:                      # noqa: BLE001
            if cam_ctor is EigenCAM:
                print(f"  ! CAM failed for {Path(row['path']).name}: {exc}")
                continue
            print(f"  ! {method_name} failed ({type(exc).__name__}: {exc})")
            print("    falling back to EigenCAM (gradient-free) for this model.")
            method_name, cam_ctor, fallback_used = "EigenCAM", EigenCAM, True
            with cam_ctor(model=model, target_layers=target_layers) as cam:
                grayscale = cam(input_tensor=tensor, targets=target)[0]

        # CAM resolution follows the feature map; resize to the display image.
        if grayscale.shape != rgb.shape[:2]:
            grayscale = np.asarray(
                Image.fromarray((grayscale * 255).astype(np.uint8)).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0

        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
        tag = "correct" if row["correct"] else "wrong"
        name = f"{tag}_{row['true_class']}_as_{row['pred_class']}_{Path(row['path']).stem}.png"
        path = out_dir / name[:150]
        _triptych(rgb, grayscale, overlay, row, method_name, mcfg["display_name"], path)
        written.append(str(path))
        print(f"  {tag:8s} {row['true_class']} -> {row['pred_class']} ({row['confidence']:.3f})")

    summary = {
        "model": model_key,
        "display_name": mcfg["display_name"],
        "method": method_name,
        "fallback_used": fallback_used,
        "target_layer": str(type(target_layers[0]).__name__),
        "images": len(written),
        "files": written,
        "note": (
            "EigenCAM substituted for Grad-CAM: gradient hooks are not supported on this "
            "model's graph. EigenCAM uses the principal component of the target layer "
            "activations and needs no gradients."
            if fallback_used else
            "Standard Grad-CAM on the last spatial convolutional block."
        ),
    }
    (out_dir / "gradcam_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{len(written)} visualizations -> {out_dir}  (method: {method_name})")
    return summary
