"""Repeat the whole pipeline under several seeds and report mean +/- std.

Why this exists: a single split produced YOLO11s-cls at macro-F1 0.6171 and
ResNet-50 at 0.6167 -- a gap of 0.0004. A gap that small is indistinguishable
from split luck, so ranking the models on it is not defensible. This script
re-runs training and evaluation under different seeds and reports the spread,
which turns "model A beat model B" into a statement about whether the difference
is larger than the run-to-run noise.

Each seed changes BOTH the train/val/test split and the training seed, because
split luck dominates variance on a dataset this small.

    python scripts/run_seeds.py                     # seeds from config + 1337, 2024
    python scripts/run_seeds.py --seeds 42 1337 2024 7
    python scripts/run_seeds.py --aggregate-only    # just rebuild the summary
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402

SCRIPTS = Path(__file__).parent

# Metrics worth aggregating. (key, label, higher_is_better)
AGG = [
    ("f1_macro", "Macro F1", True),
    ("accuracy", "Accuracy", True),
    ("balanced_accuracy", "Balanced Acc", True),
    ("precision_macro", "Macro Precision", True),
    ("recall_macro", "Macro Recall", True),
    ("roc_auc_macro", "Macro ROC-AUC", True),
    ("mcc", "MCC", True),
]


def archive_seed(cfg: dict, seed: int) -> int:
    """Copy the current results/<model>/metrics.json into results/seeds/seed<N>/."""
    dest = cfg["paths"]["results"] / "seeds" / f"seed{seed}"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for key in cfg["models"]:
        src = cfg["paths"]["results"] / key / "metrics.json"
        if src.exists():
            shutil.copy2(src, dest / f"{key}.json")
            n += 1
    return n


def clear_run_state(cfg: dict) -> None:
    """Remove checkpoints and per-model results so the next seed retrains."""
    for key in cfg["models"]:
        shutil.rmtree(cfg["paths"]["checkpoints"] / key, ignore_errors=True)
        shutil.rmtree(cfg["paths"]["results"] / key, ignore_errors=True)
    shutil.rmtree(cfg["paths"]["logs"] / "ultralytics", ignore_errors=True)


def run_one_seed(seed: int, cfg: dict, epochs: int | None) -> bool:
    """prepare -> train -> evaluate under one seed. Returns success."""
    env = dict(os.environ, HK_SEED=str(seed), PYTHONIOENCODING="utf-8")
    print(f"\n{'#' * 70}\n#  SEED {seed}\n{'#' * 70}")

    steps = [
        ([sys.executable, str(SCRIPTS / "prepare_dataset.py"), "--force"], "prepare"),
        ([sys.executable, str(SCRIPTS / "train_all.py")] + (["--epochs", str(epochs)] if epochs else []), "train"),
        ([sys.executable, str(SCRIPTS / "evaluate_all.py")], "evaluate"),
    ]
    for cmd, label in steps:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, env=env)
        print(f"[seed {seed}] {label}: exit {result.returncode} "
              f"in {(time.perf_counter() - t0) / 60:.1f} min")
        if result.returncode != 0 and label != "train":
            # train_all.py returns 0 even when one model fails, by design.
            print(f"[seed {seed}] {label} failed; skipping this seed.")
            return False
    return True


def aggregate(cfg: dict) -> pd.DataFrame | None:
    """Build the mean +/- std table across all archived seeds."""
    seed_root = cfg["paths"]["results"] / "seeds"
    if not seed_root.exists():
        return None

    records = []
    for seed_dir in sorted(seed_root.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        for f in seed_dir.glob("*.json"):
            m = json.loads(f.read_text(encoding="utf-8"))
            row = {"seed": seed, "key": f.stem, "Model": m["display_name"]}
            row.update({label: m.get(k) for k, label, _ in AGG})
            row["Parameters (M)"] = m.get("params_millions")
            row["Inference (ms)"] = m.get("inference_ms_mean")
            row["Training (min)"] = m.get("training_minutes")
            records.append(row)

    if not records:
        return None
    raw = pd.DataFrame(records)
    raw.to_csv(seed_root / "all_seed_results.csv", index=False)

    labels = [label for _, label, _ in AGG]
    grouped = raw.groupby(["key", "Model"], sort=False)
    summary = grouped[labels].agg(["mean", "std", "count"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    for extra in ("Parameters (M)", "Inference (ms)", "Training (min)"):
        summary[extra] = grouped[extra].mean().to_numpy()

    return summary.sort_values("Macro F1_mean", ascending=False).reset_index(drop=True)


def significance_note(summary: pd.DataFrame) -> str:
    """State plainly whether the top-two gap exceeds the run-to-run noise."""
    if len(summary) < 2:
        return "_Only one model; no comparison possible._"

    a, b = summary.iloc[0], summary.iloc[1]
    gap = a["Macro F1_mean"] - b["Macro F1_mean"]
    sa = 0.0 if pd.isna(a["Macro F1_std"]) else a["Macro F1_std"]
    sb = 0.0 if pd.isna(b["Macro F1_std"]) else b["Macro F1_std"]
    pooled = float(np.sqrt((sa**2 + sb**2) / 2))
    n = int(a["Macro F1_count"])

    lines = [
        f"Across **{n} seed{'s' if n != 1 else ''}** "
        f"(each seed re-splits the data and re-trains every model):", "",
        f"- Top model: **{a['Model']}**, macro-F1 {a['Macro F1_mean']:.4f} ± {sa:.4f}",
        f"- Runner-up: **{b['Model']}**, macro-F1 {b['Macro F1_mean']:.4f} ± {sb:.4f}",
        f"- Gap: {gap:.4f} · pooled standard deviation: {pooled:.4f}", "",
    ]
    if n < 2:
        lines.append("**Only one seed completed — no variance estimate is possible.**")
    elif gap > 2 * pooled:
        lines.append(
            f"The gap is more than twice the run-to-run spread, so **{a['Model']} is "
            f"genuinely ahead** of {b['Model']} on macro-F1 rather than ahead by luck.")
    elif gap > pooled:
        lines.append(
            f"The gap exceeds one standard deviation but not two. **{a['Model']} is "
            f"probably ahead**, but with {n} seeds this is suggestive, not conclusive.")
    else:
        lines.append(
            f"**The gap is smaller than the run-to-run spread.** {a['Model']} and "
            f"{b['Model']} are not distinguishable on macro-F1 from this evidence — "
            f"the ordering between them could flip on another split. Choose between "
            f"them on the criteria that _are_ clearly separated: parameter count, "
            f"model size and inference speed.")
    lines += ["", "This is why the headline ranking should be quoted with its spread, "
                  "not as a single decimal."]
    return "\n".join(lines)


def write_outputs(cfg: dict, summary: pd.DataFrame) -> None:
    out_dir = cfg["paths"]["results"] / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "model_comparison_multiseed.csv", index=False)

    md = ["| Model | Macro F1 | Accuracy | Macro ROC-AUC | MCC | Seeds |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        def pm(label):
            mean, std = r[f"{label}_mean"], r[f"{label}_std"]
            if pd.isna(mean):
                return "n/a"
            return f"{mean:.4f} ± {0.0 if pd.isna(std) else std:.4f}"
        md.append(f"| {r['Model']} | **{pm('Macro F1')}** | {pm('Accuracy')} | "
                  f"{pm('Macro ROC-AUC')} | {pm('MCC')} | {int(r['Macro F1_count'])} |")
    md += ["", significance_note(summary)]
    (out_dir / "model_comparison_multiseed.md").write_text("\n".join(md), encoding="utf-8")

    from common.plots import plot_seed_comparison
    plot_seed_comparison(summary, cfg["paths"]["plots"] / "model_comparison" / "multiseed.png")
    plot_seed_comparison(summary, out_dir / "model_comparison_multiseed.png")

    print("\n" + "\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds to run (default: config seed, 1337, 2024)")
    ap.add_argument("--epochs", type=int, help="override epochs (for a quick trial)")
    ap.add_argument("--aggregate-only", action="store_true", help="rebuild the summary, run nothing")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    # Explicit list, not derived from cfg["seed"], so re-running after the sweep
    # (which rewrites cfg["seed"] to the last seed) doesn't silently change the set.
    seeds = args.seeds or [42, 1337, 2024]

    if not args.aggregate_only:
        # The completed run in results/<model>/ belongs to the config seed. Keep it
        # rather than spending an hour reproducing it.
        base_seed = cfg["seed"]
        archived = cfg["paths"]["results"] / "seeds" / f"seed{base_seed}"
        if not archived.exists():
            n = archive_seed(cfg, base_seed)
            if n:
                print(f"Archived the existing completed run as seed {base_seed} ({n} models).")

        start = time.perf_counter()
        for seed in seeds:
            if (cfg["paths"]["results"] / "seeds" / f"seed{seed}").exists():
                print(f"\n[skip] seed {seed}: already archived")
                continue
            clear_run_state(cfg)
            if run_one_seed(seed, cfg, args.epochs):
                n = archive_seed(cfg, seed)
                print(f"[seed {seed}] archived {n} model result(s)")
        print(f"\nAll seeds finished in {(time.perf_counter() - start) / 60:.1f} min")

    summary = aggregate(cfg)
    if summary is None:
        raise SystemExit("No seed results found.")
    write_outputs(cfg, summary)

    # The sweep leaves the LAST seed's split and checkpoints on disk, but
    # gradcam/, comparison/ and best_model/ still describe whichever run came
    # before. Regenerating them here keeps every shipped artifact consistent with
    # the checkpoints that are actually on disk -- otherwise the report quotes one
    # seed's table beside another seed's confusion matrices.
    if not args.aggregate_only:
        last = seeds[-1]
        (cfg["paths"]["results"] / "ARTIFACT_SEED.txt").write_text(
            f"Single-split artifacts (checkpoints, confusion matrices, ROC curves,\n"
            f"Grad-CAM, comparison table, best-model selection) were produced with\n"
            f"seed {last}, the last seed of the sweep.\n\n"
            f"Seeds swept for the mean +/- std results: {seeds}\n"
            f"config.yaml `seed:` must equal {last} to reproduce these artifacts.\n",
            encoding="utf-8")
        print("\nRegenerating single-split artifacts from the surviving checkpoints ...")
        for script in ("generate_gradcam.py", "compare_models.py", "select_best.py"):
            subprocess.run([sys.executable, str(SCRIPTS / script)])

    print("\nRegenerating the final report with the multi-seed section ...")
    subprocess.run([sys.executable, str(SCRIPTS / "generate_report.py")])


if __name__ == "__main__":
    main()
