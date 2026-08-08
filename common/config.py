"""Config loading and path resolution. Every script starts here."""

from __future__ import annotations

import os
import random
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Read config.yaml, resolve all paths to absolute, create output dirs."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # HK_SEED lets scripts/run_seeds.py re-run the whole pipeline under a
    # different seed without editing config.yaml. It changes both the split and
    # the training seed, because split luck is the larger variance source.
    if "HK_SEED" in os.environ:
        cfg["seed"] = int(os.environ["HK_SEED"])

    # HK_DATA_ROOT env var wins over the file, so you can relocate data without
    # editing the config that ships to your senior.
    data_root = Path(os.environ.get("HK_DATA_ROOT", cfg["paths"]["data_root"]))
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()

    cfg["paths"] = {
        "project_root": PROJECT_ROOT,
        "data_root": data_root,
        "raw": data_root / "raw",
        "train": data_root / "train",
        "val": data_root / "val",
        "test": data_root / "test",
        "manifest": data_root / "manifest.csv",
        "split_report": data_root / "split_report.json",
        "checkpoints": PROJECT_ROOT / "checkpoints",
        "results": PROJECT_ROOT / "results",
        "plots": PROJECT_ROOT / "plots",
        "gradcam": PROJECT_ROOT / "gradcam",
        "logs": PROJECT_ROOT / "logs",
        # V.2: the extra archives and everything derived from them.
        "archives": data_root / "archives",
        "raw_unlabeled": data_root / "raw_unlabeled",
        "raw_videos": data_root / "raw_videos",
        "raw_segmented": data_root / "raw_segmented",
        "frames": data_root / "frames",
        "v1_manifest": data_root / cfg.get("baseline", {}).get("v1_manifest", "v1_manifest.csv"),
    }
    # Baseline checkpoint/metrics live in the repo, not the data root.
    for key in ("checkpoint", "metrics"):
        if "baseline" in cfg and key in cfg["baseline"]:
            cfg["paths"][f"baseline_{key}"] = PROJECT_ROOT / cfg["baseline"][key]
    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create every output directory the pipeline writes into."""
    p = cfg["paths"]
    model_names = list(cfg["models"].keys())
    dirs = [
        p["data_root"], p["raw"], p["logs"], p["archives"],
        p["results"] / "comparison", p["results"] / "best_model",
        p["plots"] / "training_curves", p["plots"] / "confusion_matrices",
        p["plots"] / "roc_curves", p["plots"] / "model_comparison",
    ]
    for name in model_names:
        dirs += [p["checkpoints"] / name, p["results"] / name, p["gradcam"] / name]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. Called by all training entry points."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def class_names(cfg: dict) -> list[str]:
    """The canonical class ordering: sorted directory names in the train split.

    This exact ordering is what torchvision ImageFolder AND Ultralytics both use,
    which is why probability vectors from all four models are directly comparable.
    """
    train_dir = cfg["paths"]["train"]
    if not train_dir.exists():
        raise FileNotFoundError(
            f"{train_dir} does not exist. Run: python scripts/prepare_dataset.py"
        )
    names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if not names:
        raise RuntimeError(f"No class folders inside {train_dir}")
    return names
