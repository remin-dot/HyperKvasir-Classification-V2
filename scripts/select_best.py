"""Select the best model from measured results.

Deliberately NOT accuracy-ranked. At 191:1 class imbalance the highest-accuracy
model can be the one that ignores every rare class, so accuracy carries 0.20 of
the weight while macro-F1 carries 0.40.

Each criterion is min-max normalized across the evaluated models, then weighted.
The full normalized table is printed into best_model.txt so the decision can be
audited rather than taken on trust.

    python scripts/select_best.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402
from common.plots import plot_best_model  # noqa: E402

# criterion -> (metrics.json key, higher_is_better)
CRITERIA = {
    "macro_f1": ("f1_macro", True),
    "accuracy": ("accuracy", True),
    "roc_auc_macro": ("roc_auc_macro", True),
    "pr_balance": (None, True),            # derived below
    "speed": ("inference_ms_mean", False),
    "size": ("checkpoint_size_mb", False),
}


def _raw_value(criterion: str, m: dict) -> float | None:
    if criterion == "pr_balance":
        p, r = m.get("precision_macro"), m.get("recall_macro")
        return None if p is None or r is None else 1.0 - abs(p - r)
    key, _ = CRITERIA[criterion]
    return m.get(key)


def _normalize(values: list[float], higher_is_better: bool) -> list[float]:
    """Min-max to [0, 1]. All-equal collapses to 1.0 for everyone, which cannot
    change the ranking."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    return [(hi - v) / (hi - lo) for v in values]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    weights = cfg["selection"]["weights"]
    out_dir = cfg["paths"]["results"] / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for key in cfg["models"]:
        path = cfg["paths"]["results"] / key / "metrics.json"
        if path.exists():
            models[key] = json.loads(path.read_text(encoding="utf-8"))
    if not models:
        raise SystemExit("No metrics.json files found. Run: python scripts/evaluate_all.py")
    if len(models) == 1:
        print("WARNING: only one model evaluated -- 'best' is trivially that model.")

    keys = list(models)
    raw = {c: [_raw_value(c, models[k]) for k in keys] for c in CRITERIA}

    # A criterion nobody could measure (e.g. AUC undefined) is dropped and its
    # weight redistributed, rather than silently scored as zero.
    usable = {c: v for c, v in raw.items() if all(x is not None for x in v)}
    dropped = [c for c in raw if c not in usable]
    total_weight = sum(weights[c] for c in usable)
    if not usable:
        raise SystemExit("No criterion could be measured for all models.")

    normalized = {c: _normalize(v, CRITERIA[c][1]) for c, v in usable.items()}
    contributions = {c: [n * weights[c] / total_weight for n in normalized[c]]
                     for c in usable}
    totals = [sum(contributions[c][i] for c in usable) for i in range(len(keys))]

    scores = pd.DataFrame({"Model": [models[k]["display_name"] for k in keys]})
    for c in usable:
        scores[c] = contributions[c]
    scores["Total"] = totals
    scores = scores.sort_values("Total", ascending=False).reset_index(drop=True)

    best_i = max(range(len(keys)), key=lambda i: totals[i])

    # Macro-F1 guard. A weighted sum over min-max normalized criteria can let
    # speed and size outvote a real macro-F1 advantage, which would contradict
    # the stated priority (F1 first). If that happens by more than f1_guard,
    # the higher-F1 model wins and the override is recorded.
    guard = cfg["selection"].get("f1_guard", 0.02)
    best_f1_i = max(range(len(keys)), key=lambda i: models[keys[i]]["f1_macro"])
    f1_gap = models[keys[best_f1_i]]["f1_macro"] - models[keys[best_i]]["f1_macro"]
    override = None
    if best_f1_i != best_i and f1_gap > guard:
        override = {
            "weighted_winner": models[keys[best_i]]["display_name"],
            "weighted_score": round(totals[best_i], 6),
            "overridden_by": models[keys[best_f1_i]]["display_name"],
            "f1_gap": round(f1_gap, 6),
            "guard": guard,
        }
        print(f"\nF1 GUARD TRIGGERED: {override['weighted_winner']} won the weighted score, "
              f"but {override['overridden_by']} has {f1_gap:.4f} higher macro-F1 "
              f"(guard {guard}). Selecting the higher-F1 model.\n")
        best_i = best_f1_i

    best_key = keys[best_i]
    best = models[best_key]

    # ---- best_model.txt -------------------------------------------------
    lines = [
        f"Best Model: {best['display_name']}",
        "",
        "Reason:",
    ]
    ranked = sorted(range(len(keys)), key=lambda i: totals[i], reverse=True)
    runner_up = next((i for i in ranked if i != best_i), None)

    if override:
        reason = [
            f"{best['display_name']} achieved the highest macro-F1 "
            f"({best['f1_macro']:.4f}) of every model evaluated.",
            "",
            f"On the raw weighted score, {override['weighted_winner']} edged ahead "
            f"({override['weighted_score']:.4f} vs {totals[best_i]:.4f}) on the "
            "secondary criteria",
            f"(inference speed and model size). That lead did not survive the macro-F1 "
            f"guard: the",
            f"macro-F1 difference of {override['f1_gap']:.4f} exceeds the "
            f"{override['guard']} tolerance set in",
            "config.yaml, and macro-F1 is the primary criterion for this task. The "
            "higher-F1 model",
            "was therefore selected. Both models' full numbers are below so the "
            "trade-off is visible.",
            "",
        ]
    else:
        reason = [
            f"{best['display_name']} scored highest ({totals[best_i]:.4f}) on the weighted",
            "selection criteria defined in config.yaml, which prioritise macro-F1 over raw",
            "accuracy because the HyperKvasir class distribution is imbalanced roughly 191:1.",
            "",
        ]
    reason += [
        f"Measured on the held-out test set ({best['n_test_images']} images, "
        f"{best['n_classes']} classes):",
        f"  macro F1        {best['f1_macro']:.4f}",
        f"  accuracy        {best['accuracy']:.4f}",
        f"  balanced acc.   {best['balanced_accuracy']:.4f}",
        f"  macro precision {best['precision_macro']:.4f}",
        f"  macro recall    {best['recall_macro']:.4f}",
        f"  macro ROC-AUC   " + (f"{best['roc_auc_macro']:.4f}"
                                 if best.get("roc_auc_macro") is not None else "n/a"),
        f"  MCC             {best['mcc']:.4f}",
        f"  parameters      {best['params_millions']} M",
        f"  checkpoint size {best['checkpoint_size_mb']} MB",
        f"  inference       {best['inference_ms_mean']:.2f} ms/image (batch=1, {best['fps']} FPS)",
        f"  training time   {best.get('training_minutes', 'n/a')} min",
    ]
    if runner_up is not None:
        ru = models[keys[runner_up]]
        reason += [
            "",
            f"Runner-up: {ru['display_name']} ({totals[runner_up]:.4f}), "
            f"macro F1 {ru['f1_macro']:.4f} vs {best['f1_macro']:.4f} "
            f"(difference {best['f1_macro'] - ru['f1_macro']:+.4f}).",
        ]
    lines += reason

    lines += ["", "-" * 70, "Weighted criterion contributions (min-max normalized, then weighted):", ""]
    lines.append(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    lines += ["", "Weights used:"]
    for c in usable:
        lines.append(f"  {c:<16} {weights[c] / total_weight:.3f}")
    if dropped:
        lines += ["", f"Criteria dropped (not measurable for every model): {dropped}.",
                  "Their weight was redistributed proportionally over the remaining criteria."]
    if override:
        lines += [
            "", f"NOTE: the macro-F1 guard ({override['guard']}) overrode the weighted ranking.",
            f"      Weighted winner : {override['weighted_winner']} "
            f"({override['weighted_score']:.4f})",
            f"      Selected instead: {override['overridden_by']} "
            f"(macro-F1 higher by {override['f1_gap']:.4f})",
            "      Set selection.f1_guard to 1.0 in config.yaml to use the pure weighted sum.",
        ]

    lines += ["", "-" * 70, "Raw values behind the normalization:", ""]
    raw_table = pd.DataFrame({"Model": [models[k]["display_name"] for k in keys]})
    for c in usable:
        raw_table[c] = raw[c]
    lines.append(raw_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lines += ["", "-" * 70,
              f"Checkpoint: {best['checkpoint_path']}",
              f"Copy       : {out_dir / 'best_model_checkpoint'}", ""]

    (out_dir / "best_model.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "best_model_metrics.json").write_text(json.dumps({
        "best_model_key": best_key,
        "best_model_name": best["display_name"],
        "selection_score": round(totals[best_i], 6),
        "selection_weights": {c: round(weights[c] / total_weight, 4) for c in usable},
        "criteria_dropped": dropped,
        "f1_guard": guard,
        "f1_guard_override": override,
        "selected_by": "macro-F1 guard" if override else "weighted score",
        "ranking": [{"model": models[keys[i]]["display_name"], "score": round(totals[i], 6),
                     "f1_macro": round(models[keys[i]]["f1_macro"], 6),
                     "selected": i == best_i}
                    for i in ranked],
        "metrics": best,
    }, indent=2), encoding="utf-8")
    scores.to_csv(out_dir / "selection_scores.csv", index=False)

    plot_best_model(scores, best["display_name"], out_dir / "best_model_comparison.png")

    dst = out_dir / "best_model_checkpoint"
    dst.mkdir(parents=True, exist_ok=True)
    src_dir = cfg["paths"]["checkpoints"] / best_key
    for item in src_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)
    (dst / "SOURCE.txt").write_text(
        f"{best['display_name']} ({best_key})\ncopied from {src_dir}\n"
        f"test macro-F1 {best['f1_macro']:.4f}\n", encoding="utf-8")

    print("\n".join(lines[:len(reason) + 3]))
    print(f"\n-> {out_dir / 'best_model.txt'}")
    print("Next: python scripts/generate_report.py")


if __name__ == "__main__":
    main()
