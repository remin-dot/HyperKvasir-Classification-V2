"""Single source of truth for every metric in this project.

All four models are scored by this one function on the identical test set. That
is what makes the comparison in FINAL_REPORT.md meaningful -- nothing here is
ever computed per-model or typed in by hand.

Headline metric is MACRO-F1, not accuracy. With 191:1 class imbalance a model
that predicts only the 9 largest classes and never predicts the tail still scores
~0.85 accuracy while being clinically useless.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_all(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    """Full metric bundle from ground truth and an [N, C] probability matrix."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_classes = len(class_names)

    if y_prob.shape != (len(y_true), n_classes):
        raise ValueError(f"y_prob shape {y_prob.shape} != ({len(y_true)}, {n_classes})")

    y_pred = y_prob.argmax(axis=1)

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f_cls, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(n_classes), zero_division=0)

    out = {
        "n_test_images": int(len(y_true)),
        "n_classes": n_classes,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            name: {
                "precision": float(p_cls[i]), "recall": float(r_cls[i]),
                "f1": float(f_cls[i]), "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(n_classes)).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(n_classes), target_names=class_names,
            digits=4, zero_division=0),
    }
    out.update(_auc_metrics(y_true, y_prob, n_classes))
    return out


def _auc_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> dict:
    """One-vs-rest ROC-AUC and PR-AUC.

    A class with zero test samples has an undefined AUC. Rather than crashing or
    silently substituting 0, those classes are excluded and counted, so the
    report can say exactly how many contributed.
    """
    y_bin = np.zeros((len(y_true), n_classes), dtype=np.int8)
    y_bin[np.arange(len(y_true)), y_true] = 1
    present = [i for i in range(n_classes) if 0 < y_bin[:, i].sum() < len(y_true)]

    if len(present) < 2:
        return {"roc_auc_macro": None, "pr_auc_macro": None,
                "auc_classes_used": len(present), "auc_note": "too few evaluable classes"}

    roc = roc_auc_score(y_bin[:, present], y_prob[:, present], average="macro")
    pr = average_precision_score(y_bin[:, present], y_prob[:, present], average="macro")
    note = ("all classes evaluable" if len(present) == n_classes
            else f"{n_classes - len(present)} class(es) excluded: absent from the test split")
    return {
        "roc_auc_macro": float(roc),
        "pr_auc_macro": float(pr),
        "auc_classes_used": len(present),
        "auc_note": note,
    }


def epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Cheap subset used inside the training loop for early stopping."""
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": float(f_macro)}


def summary_line(name: str, m: dict) -> str:
    auc = f"{m['roc_auc_macro']:.4f}" if m.get("roc_auc_macro") is not None else "n/a"
    return (f"{name:<20} acc={m['accuracy']:.4f}  macroF1={m['f1_macro']:.4f}  "
            f"macroP={m['precision_macro']:.4f}  macroR={m['recall_macro']:.4f}  AUC={auc}")
