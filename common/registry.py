"""Model adapters -- the piece that makes a four-way comparison legitimate.

Every model, whatever its framework, is reduced to the same tiny interface:

    predict_proba(paths) -> ndarray[N, n_classes]     (rows sum to 1)
    forward_tensor(x)    -> raw logits/probs          (for latency timing)
    cam_target_layers()  -> list[nn.Module]           (for Grad-CAM)

evaluate_all.py knows nothing else about them. Latency is measured identically
for all four: a single pre-processed batch-1 tensor through the raw forward pass,
excluding disk I/O and framework post-processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common.data import IMAGENET_MEAN, IMAGENET_STD, build_transforms


# --------------------------------------------------------------------------
# Torch models (ResNet-50, EfficientNetV2-S)
# --------------------------------------------------------------------------
def build_torch_model(model_cfg: dict, n_classes: int, pretrained: bool = True):
    """Construct an untrained-or-pretrained backbone with a fresh 23-way head."""
    source = model_cfg["source"]

    if source == "torchvision":
        from torchvision import models as tvm
        arch = model_cfg["arch"]
        weights = "IMAGENET1K_V2" if pretrained else None
        model = getattr(tvm, arch)(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features, n_classes),
        )
        return model

    if source == "timm":
        import timm
        return timm.create_model(
            model_cfg["arch"], pretrained=pretrained, num_classes=n_classes,
            drop_rate=0.3, drop_path_rate=0.2,
        )

    raise ValueError(f"Unknown torch source: {source!r}")


def split_param_groups(model, source: str, lr_head: float, lr_backbone: float, weight_decay: float):
    """Discriminative learning rates: a fresh head needs a much larger LR than a
    pretrained backbone, which we only want to nudge."""
    if source == "torchvision":
        head_params = list(model.fc.parameters())
        head_ids = {id(p) for p in head_params}
    else:
        head_params = list(model.get_classifier().parameters())
        head_ids = {id(p) for p in head_params}

    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay},
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
    ]


class _PathDataset(torch.utils.data.Dataset):
    """Dataset over an explicit, ordered list of file paths."""

    def __init__(self, paths: list[str], transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


class TorchAdapter:
    """Wraps ResNet-50 / EfficientNetV2-S."""

    def __init__(self, name: str, cfg: dict, classes: list[str], device):
        self.name = name
        self.cfg = cfg
        self.model_cfg = cfg["models"][name]
        self.display_name = self.model_cfg["display_name"]
        self.classes = classes
        self.device = device
        self.img_size = self.model_cfg["img_size"]
        self.ckpt_path = cfg["paths"]["checkpoints"] / name / "best.pt"

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.ckpt_path}. Run: python models/{name}/train.py")

        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        saved_classes = ckpt.get("classes")
        if saved_classes and list(saved_classes) != list(classes):
            raise RuntimeError(
                f"{name}: checkpoint class order does not match the dataset.\n"
                f"  checkpoint: {saved_classes}\n  dataset   : {classes}\n"
                f"Re-run prepare_dataset.py and retrain.")

        self.model = build_torch_model(self.model_cfg, len(classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(device)
        self.transform = build_transforms(self.img_size, train=False)
        self.train_meta = ckpt.get("meta", {})

    @torch.inference_mode()
    def predict_proba(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        loader = torch.utils.data.DataLoader(
            _PathDataset(paths, self.transform), batch_size=batch_size,
            shuffle=False, num_workers=self.cfg["training"]["num_workers"],
            pin_memory=self.device.type == "cuda",
        )
        out = []
        for batch in loader:
            logits = self.model(batch.to(self.device, non_blocking=True))
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def forward_tensor(self, x: torch.Tensor):
        return self.model(x)

    def sample_tensor(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")).unsqueeze(0).to(self.device)

    def cam_model(self):
        return self.model

    def cam_target_layers(self):
        if self.model_cfg["source"] == "torchvision":
            return [self.model.layer4[-1]]
        return [self.model.blocks[-1]]        # timm EfficientNetV2 last stage

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# --------------------------------------------------------------------------
# Ultralytics models (YOLOv8-cls, YOLO11-cls)
# --------------------------------------------------------------------------
class UltralyticsAdapter:
    """Wraps YOLOv8s-cls / YOLO11s-cls.

    Inference goes through the underlying nn.Module rather than YOLO.predict(),
    so latency is measured on the same footing as the torch models. Preprocessing
    uses Ultralytics' own classify_transforms when available, so accuracy still
    reflects how the model was actually trained.
    """

    def __init__(self, name: str, cfg: dict, classes: list[str], device):
        from ultralytics import YOLO

        self.name = name
        self.cfg = cfg
        self.model_cfg = cfg["models"][name]
        self.display_name = self.model_cfg["display_name"]
        self.classes = classes
        self.device = device
        self.img_size = self.model_cfg["img_size"]
        self.ckpt_path = cfg["paths"]["checkpoints"] / name / "best.pt"

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {self.ckpt_path}. Run: python models/{name}/train.py")

        self.yolo = YOLO(str(self.ckpt_path))
        self.model = self.yolo.model.to(device).eval().float()

        # The comparison assumes identical class indices across all four models.
        # Ultralytics sorts class folders exactly like ImageFolder, but assert it
        # rather than trust it -- a silent mismatch would corrupt every metric.
        yolo_names = [self.yolo.names[i] for i in sorted(self.yolo.names)]
        if yolo_names != list(classes):
            raise RuntimeError(
                f"{name}: Ultralytics class order differs from the dataset.\n"
                f"  ultralytics: {yolo_names}\n  dataset    : {classes}")

        self.transform = self._build_transform()

    def _build_transform(self):
        """Ultralytics' eval transform if we can import it, else our own."""
        try:
            from ultralytics.data.augment import classify_transforms
            return classify_transforms(size=self.img_size)
        except Exception:                      # noqa: BLE001 - API moved between versions
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    @staticmethod
    def _to_proba(out) -> torch.Tensor:
        """Ultralytics' Classify head has returned raw logits in some versions and
        softmaxed scores in others. Detect which, instead of guessing."""
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.float()
        sums = out.sum(dim=1)
        already_proba = bool(torch.all(out >= 0)) and bool(torch.allclose(
            sums, torch.ones_like(sums), atol=1e-3))
        return out if already_proba else torch.softmax(out, dim=1)

    @torch.inference_mode()
    def predict_proba(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        loader = torch.utils.data.DataLoader(
            _PathDataset(paths, self.transform), batch_size=batch_size,
            shuffle=False, num_workers=self.cfg["training"]["num_workers"],
            pin_memory=self.device.type == "cuda",
        )
        out = []
        for batch in loader:
            out.append(self._to_proba(self.model(batch.to(self.device, non_blocking=True))).cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.inference_mode()
    def forward_tensor(self, x: torch.Tensor):
        return self.model(x)

    def sample_tensor(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")).unsqueeze(0).to(self.device)

    def cam_model(self):
        return self.model

    def cam_target_layers(self):
        """Last convolutional block before the Classify head.

        model.model is an nn.Sequential; [-1] is the Classify head itself (pool +
        linear, spatially collapsed and useless for a heatmap), so [-2] is the
        deepest layer that still has spatial extent.
        """
        seq = self.model.model
        return [seq[-2]]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


# --------------------------------------------------------------------------
def load_model(name: str, cfg: dict, classes: list[str], device):
    """Build the right adapter for a model key from config.yaml."""
    if name not in cfg["models"]:
        raise KeyError(f"Unknown model {name!r}. Known: {list(cfg['models'])}")
    source = cfg["models"][name]["source"]
    if source == "ultralytics":
        return UltralyticsAdapter(name, cfg, classes, device)
    return TorchAdapter(name, cfg, classes, device)


def model_names(cfg: dict) -> list[str]:
    return list(cfg["models"].keys())
