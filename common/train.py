"""Dispatch to the right trainer for a model key."""

from __future__ import annotations


def train_model(model_key: str, cfg: dict, epochs: int | None = None) -> dict:
    source = cfg["models"][model_key]["source"]
    if source == "ultralytics":
        from common.yolo_trainer import train_yolo_model
        return train_yolo_model(model_key, cfg, epochs)
    from common.torch_trainer import train_torch_model
    return train_torch_model(model_key, cfg, epochs)
