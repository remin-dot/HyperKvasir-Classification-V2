"""Build the side-by-side comparison table from results/*/metrics.json.

Every number is read from a file written by evaluate_all.py. Nothing is typed in.

    python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402
from common.plots import plot_model_comparison  # noqa: E402

# (display column, key in metrics.json, rounding)
COLUMNS = [
    ("Model", "display_name", None),
    ("Framework", "framework", None),
    ("Accuracy", "accuracy", 4),
    ("Balanced Acc", "balanced_accuracy", 4),
    ("Precision", "precision_macro", 4),
    ("Recall", "recall_macro", 4),
    ("F1", "f1_macro", 4),
    ("F1 (weighted)", "f1_weighted", 4),
    ("AUC", "roc_auc_macro", 4),
    ("PR-AUC", "pr_auc_macro", 4),
    ("MCC", "mcc", 4),
    ("Kappa", "cohen_kappa", 4),
    ("Parameters (M)", "params_millions", 2),
    ("Model Size (MB)", "checkpoint_size_mb", 2),
    ("Inference (ms)", "inference_ms_mean", 3),
    ("FPS", "fps", 1),
    ("Training (min)", "training_minutes", 2),
    ("Epochs", "epochs_run", None),
    ("Peak VRAM (MB)", "train_peak_vram_mb", 1),
]


def collect(cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Read every metrics.json; collect FAILED.json records separately."""
    rows, failures = [], []
    for key in cfg["models"]:
        result_dir = cfg["paths"]["results"] / key
        metrics_path = result_dir / "metrics.json"
        failed_path = result_dir / "FAILED.json"

        if not metrics_path.exists():
            record = (json.loads(failed_path.read_text(encoding="utf-8"))
                      if failed_path.exists()
                      else {"model_key": key,
                            "display_name": cfg["models"][key]["display_name"],
                            "stage": "unknown",
                            "error": "No metrics.json and no failure record — "
                                     "the model was never trained or evaluated."})
            failures.append(record)
            continue

        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"key": key}
        for label, field, digits in COLUMNS:
            value = m.get(field)
            row[label] = round(value, digits) if (digits and isinstance(value, (int, float))) else value
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("F1", ascending=False).reset_index(drop=True)
    return df, failures


def to_markdown(df: pd.DataFrame) -> str:
    """Headline table in the exact shape requested, plus a resource table."""
    if df.empty:
        return "_No models were successfully evaluated._"

    head = ["| Model | Accuracy | Precision | Recall | F1 | AUC | Parameters | Inference Time | Training Time |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        auc = f"{r['AUC']:.4f}" if pd.notna(r["AUC"]) else "n/a"
        head.append(
            f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | "
            f"**{r['F1']:.4f}** | {auc} | {r['Parameters (M)']:.2f} M | "
            f"{r['Inference (ms)']:.2f} ms | {r['Training (min)']:.1f} min |")

    extra = ["", "Additional measurements:", "",
             "| Model | Balanced Acc | Weighted F1 | PR-AUC | MCC | Kappa | Size (MB) | FPS | Epochs | Peak VRAM (MB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        pr = f"{r['PR-AUC']:.4f}" if pd.notna(r["PR-AUC"]) else "n/a"
        vram = f"{r['Peak VRAM (MB)']:.0f}" if pd.notna(r["Peak VRAM (MB)"]) else "n/a"
        extra.append(
            f"| {r['Model']} | {r['Balanced Acc']:.4f} | {r['F1 (weighted)']:.4f} | {pr} | "
            f"{r['MCC']:.4f} | {r['Kappa']:.4f} | {r['Model Size (MB)']:.1f} | "
            f"{r['FPS']:.0f} | {r['Epochs']} | {vram} |")
    return "\n".join(head + extra)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    out_dir = cfg["paths"]["results"] / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, failures = collect(cfg)
    if df.empty:
        raise SystemExit("No evaluated models found. Run: python scripts/evaluate_all.py")

    export = df.drop(columns=["key"])
    export.to_csv(out_dir / "model_comparison.csv", index=False)
    try:
        export.to_excel(out_dir / "model_comparison.xlsx", index=False, sheet_name="Comparison")
        print(f"-> {out_dir / 'model_comparison.xlsx'}")
    except Exception as exc:  # noqa: BLE001 - openpyxl missing shouldn't kill the run
        print(f"xlsx export skipped ({type(exc).__name__}: {exc}). CSV and PNG still written.")

    (out_dir / "model_comparison.md").write_text(to_markdown(df), encoding="utf-8")
    plot_model_comparison(df, cfg["paths"]["plots"] / "model_comparison" / "model_comparison.png")
    plot_model_comparison(df, out_dir / "model_comparison.png")

    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print(f"-> {out_dir / 'model_comparison.csv'}")
    print()
    print(export.to_string(index=False))
    if failures:
        print(f"\n{len(failures)} model(s) missing from the comparison:")
        for f in failures:
            print(f"  {f.get('display_name', f['model_key'])}: "
                  f"{f.get('error_type', '')} {f.get('error', '')[:120]}")
    print("\nNext: python scripts/select_best.py")


if __name__ == "__main__":
    main()
