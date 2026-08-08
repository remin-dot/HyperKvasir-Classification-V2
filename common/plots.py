"""All figures. Matplotlib only -- seaborn would add a dependency for styling we
can get from rcParams.

Every plot is drawn from data already written to disk by another stage, so plots
can always be regenerated without retraining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # headless: no display needed, no Tk crashes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
})


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  plot -> {out}")
    return out


def plot_training_curves(history: pd.DataFrame, title: str, out: Path) -> Path | None:
    """Loss / accuracy / macro-F1 per epoch.

    Handles the Ultralytics case, where macro-F1 per epoch simply does not exist:
    the panel is drawn with an explanatory note rather than left blank or faked.
    """
    if history is None or history.empty:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = history["epoch"]

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(x, history["train_loss"], label="train", lw=1.6)
    if "val_loss" in history:
        ax.plot(x, history["val_loss"], label="val", lw=1.6)
    ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax.legend()

    ax = axes[1]
    if "train_acc" in history:
        ax.plot(x, history["train_acc"], label="train", lw=1.6)
    if "val_acc" in history:
        ax.plot(x, history["val_acc"], label="val", lw=1.6)
    ax.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax.legend()

    ax = axes[2]
    if "val_f1_macro" in history:
        ax.plot(x, history["val_f1_macro"], color="tab:green", lw=1.8)
        best = history["val_f1_macro"].idxmax()
        ax.axvline(history.loc[best, "epoch"], ls="--", c="grey", lw=1)
        ax.set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1 (early-stopping metric)")
    else:
        ax.text(0.5, 0.5, "Per-epoch macro-F1 not reported\nby the Ultralytics trainer\n"
                          "(top-1 accuracy shown left).",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="grey")
        ax.set(title="Validation macro-F1")
        ax.set_axis_off()

    fig.suptitle(f"{title} — training curves", fontsize=12, y=1.02)
    return _save(fig, out)


def plot_confusion_matrix(cm, classes: list[str], title: str, out: Path,
                          normalize: bool = True) -> Path:
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        fmt, label = "{:.2f}", "row-normalized (recall per class)"
    else:
        shown, fmt, label = cm, "{:.0f}", "raw counts"

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.55), max(8, n * 0.5)))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() if shown.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", title=f"{title} — confusion matrix ({label})")
    ax.grid(False)

    threshold = shown.max() / 2 if shown.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            if shown[i, j] > 0:
                ax.text(j, i, fmt.format(shown[i, j]), ha="center", va="center",
                        fontsize=5.5, color="white" if shown[i, j] > threshold else "black")
    return _save(fig, out)


def plot_roc_curves(y_true, y_prob, classes: list[str], title: str, out: Path) -> Path:
    """One-vs-rest ROC per class plus the macro average.

    23 thin curves are deliberately unlabelled -- the legend would be larger than
    the plot. The macro curve is the one to read; per-class AUCs live in metrics.json.
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(7, 6.2))

    aucs, all_fpr = [], np.linspace(0, 1, 200)
    interp_tprs = []
    for i, name in enumerate(classes):
        binary = (y_true == i).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue                      # undefined ROC; excluded and reported in metrics
        fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        interp_tprs.append(np.interp(all_fpr, fpr, tpr))
        ax.plot(fpr, tpr, lw=0.8, alpha=0.35)

    if interp_tprs:
        macro_tpr = np.mean(interp_tprs, axis=0)
        ax.plot(all_fpr, macro_tpr, lw=2.6, color="crimson",
                label=f"macro average (AUC = {np.mean(aucs):.4f}, {len(aucs)}/{len(classes)} classes)")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title=f"{title} — ROC (one-vs-rest)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, out)


def plot_class_distribution(counts: pd.DataFrame, out: Path) -> Path:
    """Stacked per-split counts on a log axis.

    Log scale is not decoration: at 1148 vs 6 images a linear axis renders the
    three rarest classes as invisible slivers, which is exactly the fact the
    reader most needs to see.
    """
    counts = counts.sort_values("total", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(counts))
    bottom = np.zeros(len(counts))
    for split, color in (("train", "tab:blue"), ("val", "tab:orange"), ("test", "tab:green")):
        if split in counts:
            ax.bar(x, counts[split], bottom=bottom, label=split, color=color)
            bottom += counts[split].to_numpy()

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(counts["class"], rotation=90, fontsize=8)
    ax.set(ylabel="images (log scale)",
           title=f"HyperKvasir class distribution — {int(counts['total'].sum()):,} images, "
                 f"{len(counts)} classes, imbalance "
                 f"{counts['total'].max() / max(1, counts['total'].min()):.0f}:1")
    ax.legend()
    for i, total in enumerate(counts["total"]):
        ax.text(i, total * 1.1, str(int(total)), ha="center", fontsize=6.5)
    return _save(fig, out)


def plot_model_comparison(df: pd.DataFrame, out: Path) -> Path:
    """Four-panel comparison: headline metrics, per-model AUC, speed, size."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    names = df["Model"].tolist()
    x = np.arange(len(names))

    ax = axes[0, 0]
    width = 0.2
    for k, (col, label) in enumerate([("Accuracy", "accuracy"), ("Precision", "macro precision"),
                                      ("Recall", "macro recall"), ("F1", "macro F1")]):
        ax.bar(x + (k - 1.5) * width, df[col], width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set(ylabel="score", ylim=(0, 1), title="Test-set performance")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    auc_vals = pd.to_numeric(df["AUC"], errors="coerce").fillna(0)
    bars = ax.bar(x, auc_vals, color="tab:purple")
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set(ylabel="macro ROC-AUC", ylim=(0, 1.05), title="Macro ROC-AUC (one-vs-rest)")

    ax = axes[1, 0]
    lat = pd.to_numeric(df["Inference (ms)"], errors="coerce")
    bars = ax.bar(x, lat, color="tab:orange")
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8)
    ax.set(ylabel="ms / image (batch=1)", title="Inference latency — lower is better")

    ax = axes[1, 1]
    params = pd.to_numeric(df["Parameters (M)"], errors="coerce")
    f1 = pd.to_numeric(df["F1"], errors="coerce")
    ax.scatter(params, f1, s=140, c=range(len(names)), cmap="viridis", zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (params.iloc[i], f1.iloc[i]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8)
    ax.set(xlabel="parameters (millions)", ylabel="macro F1",
           title="Accuracy vs model size — upper-left is best")

    fig.suptitle("HyperKvasir — model comparison on the identical test set", fontsize=13)
    fig.tight_layout()
    return _save(fig, out)


def plot_seed_comparison(summary: pd.DataFrame, out: Path) -> Path:
    """Mean +/- std across seeds, with the error bars that decide whether a
    ranking is real. Overlapping bars mean the ordering is not established."""
    panels = [("Macro F1", "macro F1"), ("Accuracy", "accuracy"),
              ("Macro ROC-AUC", "macro ROC-AUC"), ("MCC", "MCC")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    names = summary["Model"].tolist()
    x = np.arange(len(names))
    n_seeds = int(summary["Macro F1_count"].max())

    for ax, (label, pretty) in zip(axes, panels):
        mean = pd.to_numeric(summary[f"{label}_mean"], errors="coerce")
        std = pd.to_numeric(summary[f"{label}_std"], errors="coerce").fillna(0.0)
        ax.bar(x, mean, yerr=std, capsize=6, color="tab:blue", alpha=0.85,
               error_kw={"ecolor": "black", "elinewidth": 1.4})
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set(ylabel=pretty, title=pretty)
        lo = max(0.0, float((mean - std).min()) - 0.05)
        hi = min(1.02, float((mean + std).max()) + 0.05)
        ax.set_ylim(lo, hi)
        for i, (m, s) in enumerate(zip(mean, std)):
            if pd.notna(m):
                ax.text(i, m + s + (hi - lo) * 0.02, f"{m:.3f}", ha="center", fontsize=7.5)

    fig.suptitle(f"Mean ± standard deviation across {n_seeds} seeds "
                 f"(each seed re-splits the data and re-trains every model)", fontsize=12)
    fig.tight_layout()
    return _save(fig, out)


def plot_best_model(scores: pd.DataFrame, best_name: str, out: Path) -> Path:
    """Stacked weighted-criterion scores showing why the winner won."""
    criteria = [c for c in scores.columns if c not in ("Model", "Total")]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(scores))
    bottom = np.zeros(len(scores))
    cmap = plt.get_cmap("tab10")

    for i, crit in enumerate(criteria):
        vals = scores[crit].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=crit, color=cmap(i))
        bottom += vals

    ax.set_xticks(x)
    labels = [f"{n}\n(WINNER)" if n == best_name else n for n in scores["Model"]]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set(ylabel="weighted score contribution",
           title=f"Best-model selection — winner: {best_name}")
    ax.legend(fontsize=8, ncol=3)
    for i, total in enumerate(scores["Total"]):
        ax.text(i, total + 0.012, f"{total:.4f}", ha="center", fontweight="bold", fontsize=9)
    return _save(fig, out)
