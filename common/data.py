"""Dataset audit, preprocessing, splitting and transforms.

The output of prepare_dataset.py is a plain ImageFolder tree:

    <data_root>/train/<class>/*.jpg
    <data_root>/val/<class>/*.jpg
    <data_root>/test/<class>/*.jpg

That single layout feeds torchvision ImageFolder AND Ultralytics classification
unchanged, so all four models see byte-identical images in an identical class
order. No per-model conversion anywhere in this project.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# Per-image preprocessing
# --------------------------------------------------------------------------
def crop_borders(img: Image.Image, threshold: int = 20) -> tuple[Image.Image, bool]:
    """Trim the black letterbox around the circular/rectangular endoscope view.

    Returns (image, was_cropped). Bails out if the detected content box is less
    than 30% of the frame -- that means detection failed (very dark image) and
    cropping would destroy real data.
    """
    gray = np.asarray(img.convert("L"))
    rows = np.where(gray.max(axis=1) > threshold)[0]
    cols = np.where(gray.max(axis=0) > threshold)[0]
    if rows.size == 0 or cols.size == 0:
        return img, False

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    if (bottom - top) * (right - left) < 0.30 * gray.size:
        return img, False
    if (top, left, bottom, right) == (0, 0, gray.shape[0], gray.shape[1]):
        return img, False
    return img.crop((left, top, right, bottom)), True


def detect_pip(img: Image.Image, border_frac: float = 0.25, min_ratio: float = 0.005) -> bool:
    """Flag the green 'ScopeGuide' picture-in-picture navigation overlay.

    Detection only -- we deliberately do NOT mask it. Under narrow-band imaging
    real mucosa goes green too, so auto-masking would delete diagnostic tissue.
    The count goes in the dataset report and Grad-CAM is the actual check that
    no model learned to read this box instead of the mucosa.
    """
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = a.shape[:2]
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))

    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :], mask[-bh:, :], mask[:, :bw], mask[:, -bw:] = True, True, True, True

    g, r, b = a[..., 1], a[..., 0], a[..., 2]
    green = (g > r + 40) & (g > b + 40) & (g > 60)
    return bool((green & mask).sum() / mask.sum() > min_ratio)


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Manifest / audit
# --------------------------------------------------------------------------
def find_images(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _inspect(path: Path, detect_pip_flag: bool) -> dict:
    """Open one image: verify it decodes, record size, hash, PiP flag."""
    rec = {
        "path": str(path), "class": path.parent.name,
        "category": path.parent.parent.name if path.parent.parent else "",
        "tract": path.parent.parent.parent.name if path.parent.parent.parent else "",
        "width": None, "height": None, "md5": None, "has_pip": False,
        "ok": False, "error": "",
    }
    try:
        with Image.open(path) as im:
            im.verify()                      # catches truncated / corrupt files
        with Image.open(path) as im:
            im = im.convert("RGB")
            rec["width"], rec["height"] = im.size
            if detect_pip_flag:
                rec["has_pip"] = detect_pip(im)
        rec["md5"] = _md5(path)
        rec["ok"] = True
    except Exception as exc:                 # noqa: BLE001 - want the message, whatever it is
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def build_manifest(raw_dir: Path, detect_pip_flag: bool = True, workers: int = 16) -> pd.DataFrame:
    """Audit every image under raw_dir. Returns one row per file."""
    paths = find_images(raw_dir)
    if not paths:
        raise RuntimeError(
            f"No images found under {raw_dir}.\n"
            f"Run: python scripts/download_dataset.py"
        )
    print(f"Auditing {len(paths):,} images with {workers} threads ...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda p: _inspect(p, detect_pip_flag), paths))
    return pd.DataFrame(records)


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop byte-identical duplicates, keeping the first occurrence.

    This is the only leakage we can fully eliminate. HyperKvasir ships no patient
    IDs, so near-duplicate frames from the same examination can still land in
    different splits -- documented, not fixable from the data we have.
    """
    before = len(df)
    df = df.drop_duplicates(subset="md5", keep="first").reset_index(drop=True)
    return df, before - len(df)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    """Class-stratified train/val/test split, guaranteeing >=1 image per class
    in every split.

    Classes like hemorrhoids (6 images) land at 4/1/1. sklearn's stratify would
    happily round one of those to zero, so tiny classes are split by hand.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed)
    df = df.copy()
    df["split"] = ""

    for cls, group in df.groupby("class", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)

        if n < 6:
            # Too few for a meaningful proportional split: 1 val, 1 test, rest train.
            n_val = n_test = 1 if n >= 3 else 0
            n_train = n - n_val - n_test
            parts = [("train", idx[:n_train]),
                     ("val", idx[n_train:n_train + n_val]),
                     ("test", idx[n_train + n_val:])]
        else:
            train_idx, rest = train_test_split(
                idx, test_size=ratios["val"] + ratios["test"], random_state=seed)
            rel_test = ratios["test"] / (ratios["val"] + ratios["test"])
            val_idx, test_idx = train_test_split(rest, test_size=rel_test, random_state=seed)
            parts = [("train", train_idx), ("val", val_idx), ("test", test_idx)]

        for split_name, part in parts:
            df.loc[part, "split"] = split_name

    assert (df["split"] != "").all(), "some rows were never assigned a split"
    return df


def materialize(df: pd.DataFrame, cfg: dict) -> dict:
    """Write the split folders.

    When crop_borders or resize_max_side is enabled we re-encode each image, so
    the preprocessing is baked into the files on disk. That matters: Ultralytics
    reads these folders directly and will not run our transform pipeline, so any
    preprocessing that is not baked in would apply to two models and not the
    other two -- silently invalidating the comparison.
    """
    pre = cfg["preprocess"]
    do_process = pre["crop_borders"] or pre["resize_max_side"]
    paths = cfg["paths"]
    stats = {"written": 0, "cropped": 0, "hardlinked": 0, "failed": 0}

    for split in ("train", "val", "test"):
        root = paths[split]
        if root.exists():
            import shutil
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> tuple[bool, bool, bool]:
        # dict rows, not itertuples: 'class' is a Python keyword and itertuples
        # silently renames it to a positional field.
        src = Path(row["path"])
        dst_dir = paths[row["split"]] / row["class"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{src.stem}.jpg"
        try:
            if not do_process:
                try:
                    import os
                    os.link(src, dst)          # instant, no extra disk
                    return True, False, True
                except OSError:
                    import shutil
                    shutil.copy2(src, dst)
                    return True, False, False
            with Image.open(src) as im:
                im = im.convert("RGB")
                cropped = False
                if pre["crop_borders"]:
                    im, cropped = crop_borders(im)
                if pre["resize_max_side"]:
                    im = resize_max_side(im, pre["resize_max_side"])
                im.save(dst, "JPEG", quality=pre["jpeg_quality"])
            return True, cropped, False
        except Exception as exc:               # noqa: BLE001
            print(f"  ! failed {src.name}: {exc}")
            return False, False, False

    rows = df.to_dict(orient="records")
    print(f"Materializing {len(rows):,} images into {paths['data_root']} ...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        for ok, cropped, linked in pool.map(one, rows):
            stats["written" if ok else "failed"] += 1
            stats["cropped"] += int(cropped)
            stats["hardlinked"] += int(linked)
    return stats


# --------------------------------------------------------------------------
# Transforms and loaders (torch models only -- Ultralytics uses its own)
# --------------------------------------------------------------------------
def build_transforms(img_size: int, train: bool):
    """torchvision transforms. No albumentations: torchvision already covers
    every augmentation this task needs, and one dependency is cheaper than two.

    Full +/-180 degree rotation is correct here -- an endoscope frame has no
    canonical 'up', unlike natural images.
    """
    from torchvision import transforms

    if not train:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_dataloaders(cfg: dict, img_size: int, batch_size: int):
    """train/val/test loaders over the materialized ImageFolder tree."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    paths = cfg["paths"]
    nw = cfg["training"]["num_workers"]
    pin = torch.cuda.is_available()

    sets = {
        "train": ImageFolder(paths["train"], build_transforms(img_size, True)),
        "val": ImageFolder(paths["val"], build_transforms(img_size, False)),
        "test": ImageFolder(paths["test"], build_transforms(img_size, False)),
    }
    # Guard the assumption the whole comparison rests on.
    assert sets["train"].classes == sets["val"].classes == sets["test"].classes, \
        "class ordering differs between splits"

    loaders = {
        name: DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=nw, pin_memory=pin, drop_last=False,
            persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
        )
        for name, ds in sets.items()
    }
    return loaders, sets["train"].classes


def class_weights(train_dir: Path, classes: list[str]):
    """Inverse-sqrt class weights, normalized to mean 1.

    Raw inverse frequency (1/n) over-corrects badly at 191:1 -- it would weight
    hemorrhoids ~191x and the model collapses onto the rare classes. sqrt is the
    standard middle ground.
    """
    import torch
    counts = np.array([len(list((train_dir / c).glob("*.jpg"))) for c in classes], dtype=np.float64)
    counts = np.maximum(counts, 1)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def test_image_paths(cfg: dict, classes: list[str]) -> tuple[list[str], np.ndarray]:
    """The canonical test set: identical file list and order for every model.

    Plain directory walk rather than ImageFolder -- this replicates ImageFolder's
    ordering (classes sorted, then filenames sorted within each class) without
    dragging torchvision into a function that only enumerates files.
    """
    test_dir = cfg["paths"]["test"]
    found = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    assert found == list(classes), f"class mismatch: {found} != {list(classes)}"

    paths: list[str] = []
    labels: list[int] = []
    for idx, cls in enumerate(classes):
        for p in sorted((test_dir / cls).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(str(p))
                labels.append(idx)
    if not paths:
        raise RuntimeError(f"No test images found under {test_dir}")
    return paths, np.array(labels, dtype=np.int64)
