"""Audit, deduplicate, split and materialize the HyperKvasir dataset.

Produces the ImageFolder tree that all four models train from, plus a dataset
report and the class-distribution figure.

    python scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config, set_seed  # noqa: E402
from common.data import build_manifest, dedupe, materialize, stratified_split  # noqa: E402
from common.plots import plot_class_distribution  # noqa: E402


def build_report(df: pd.DataFrame, stats: dict, audit: dict, cfg: dict) -> tuple[str, pd.DataFrame]:
    counts = (df.pivot_table(index="class", columns="split", values="path",
                             aggfunc="count", fill_value=0)
                .reset_index())
    for split in ("train", "val", "test"):
        if split not in counts:
            counts[split] = 0
    counts["total"] = counts[["train", "val", "test"]].sum(axis=1)
    counts = counts.sort_values("total", ascending=False).reset_index(drop=True)

    imbalance = counts["total"].max() / max(1, counts["total"].min())
    tiny = counts[counts["test"] <= 2]

    lines = [
        "# HyperKvasir Dataset Report", "",
        f"Source archive : {cfg['dataset']['url']}",
        "License        : CC BY 4.0",
        f"Data root      : `{cfg['paths']['data_root']}`", "",
        "## Audit", "",
        f"| Item | Count |", "|---|---:|",
        f"| Files found | {audit['found']:,} |",
        f"| Corrupt / unreadable | {audit['corrupt']:,} |",
        f"| Exact duplicates removed | {audit['duplicates']:,} |",
        f"| Images used | {len(df):,} |",
        f"| Classes | {len(counts)} |",
        f"| Green ScopeGuide overlay detected | {audit['pip']:,} |",
        f"| Black border cropped | {stats['cropped']:,} |",
        f"| Resolutions (distinct) | {audit['resolutions']} |", "",
        "## Split", "",
        f"Stratified {cfg['dataset']['splits']['train']:.0%} / "
        f"{cfg['dataset']['splits']['val']:.0%} / {cfg['dataset']['splits']['test']:.0%}, "
        f"seed {cfg['seed']}.", "",
        f"| Split | Images |", "|---|---:|",
        f"| train | {int(counts['train'].sum()):,} |",
        f"| val | {int(counts['val'].sum()):,} |",
        f"| test | {int(counts['test'].sum()):,} |", "",
        "## Class distribution", "",
        "| Class | Train | Val | Test | Total |", "|---|---:|---:|---:|---:|",
    ]
    for _, r in counts.iterrows():
        lines.append(f"| {r['class']} | {int(r['train'])} | {int(r['val'])} | "
                     f"{int(r['test'])} | {int(r['total'])} |")

    lines += [
        "", "## Class imbalance", "",
        f"**{imbalance:.0f}:1** between the largest and smallest class "
        f"(`{counts.iloc[0]['class']}` = {int(counts.iloc[0]['total'])}, "
        f"`{counts.iloc[-1]['class']}` = {int(counts.iloc[-1]['total'])}).", "",
        "Accuracy is therefore not a meaningful headline metric: a model that predicts",
        "only the largest classes and never predicts the tail can still score around 0.85.",
        "**Macro-F1 is the primary metric throughout this project.**", "",
    ]
    if len(tiny):
        lines += [
            f"### Classes with <=2 test images ({len(tiny)})", "",
            "Their per-class precision/recall/F1 are reported for completeness but are",
            "statistically meaningless at this sample size:", "",
        ]
        lines += [f"- `{r['class']}` — {int(r['total'])} images total, "
                  f"{int(r['test'])} in test" for _, r in tiny.iterrows()]
        lines.append("")

    lines += [
        "## Known limitations", "",
        "1. **No patient identifiers.** HyperKvasir does not ship them, so near-duplicate",
        "   frames from the same examination can fall into different splits. Byte-identical",
        "   duplicates are removed, but perceptual near-duplicates cannot be detected",
        "   reliably without patient metadata. This inflates results on this dataset",
        "   generally, including published baselines.",
        "2. **Green ScopeGuide overlay is detected, not masked.** Under narrow-band imaging",
        "   real mucosa is also green, so automatic masking would delete diagnostic tissue.",
        "   Grad-CAM output is the check that no model learned to read this overlay.",
        f"3. **Images are pre-processed once on disk** (border crop, max side "
        f"{cfg['preprocess']['resize_max_side']} px, JPEG q{cfg['preprocess']['jpeg_quality']}).",
        "   This is required for fairness: Ultralytics reads these folders directly and will",
        "   not run our transform pipeline, so preprocessing that is not baked in would apply",
        "   to two models and not the other two.", "",
    ]
    return "\n".join(lines), counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="rebuild even if splits exist")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])
    paths = cfg["paths"]

    if paths["train"].exists() and any(paths["train"].iterdir()) and not args.force:
        n = sum(1 for _ in paths["train"].rglob("*.jpg"))
        print(f"Splits already exist ({n:,} train images). Use --force to rebuild.")
        return

    print(f"Raw data : {paths['raw']}")
    df = build_manifest(paths["raw"], cfg["preprocess"]["detect_pip"])

    audit = {
        "found": len(df),
        "corrupt": int((~df["ok"]).sum()),
        "pip": int(df["has_pip"].sum()),
        "resolutions": int(df.dropna(subset=["width"])
                           .groupby(["width", "height"]).ngroups),
    }
    if audit["corrupt"]:
        print(f"WARNING: {audit['corrupt']} unreadable file(s) excluded:")
        for _, r in df[~df["ok"]].head(10).iterrows():
            print(f"  {r['path']}: {r['error']}")
    df = df[df["ok"]].reset_index(drop=True)

    duplicates = 0
    if cfg["dataset"]["dedup"]:
        df, duplicates = dedupe(df)
        print(f"Removed {duplicates} exact duplicate(s).")
    audit["duplicates"] = duplicates

    print(f"{len(df):,} usable images across {df['class'].nunique()} classes.")
    df = stratified_split(df, cfg["dataset"]["splits"], cfg["seed"])
    print(df["split"].value_counts().to_string())

    stats = materialize(df, cfg)
    print(f"Written {stats['written']:,} | cropped {stats['cropped']:,} | "
          f"hardlinked {stats['hardlinked']:,} | failed {stats['failed']:,}")

    df.to_csv(paths["manifest"], index=False)
    report, counts = build_report(df, stats, audit, cfg)

    (paths["results"] / "dataset_report.md").write_text(report, encoding="utf-8")
    (paths["project_root"] / "dataset" / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (paths["project_root"] / "dataset" / "README.md").write_text(
        f"# Dataset\n\nImages live at `{paths['data_root']}` (kept outside the project so "
        f"OneDrive does not sync 10k files and make training I/O-bound).\n\n"
        f"Override with the `HK_DATA_ROOT` environment variable or `paths.data_root` in "
        f"config.yaml.\n\n```\n{paths['data_root']}\n"
        f"  raw/      extracted archive\n  train/    <class>/*.jpg\n"
        f"  val/      <class>/*.jpg\n  test/     <class>/*.jpg\n"
        f"  manifest.csv\n```\n\nFull statistics: `results/dataset_report.md`\n",
        encoding="utf-8")

    json.dump(
        {"audit": audit, "materialize": stats, "seed": cfg["seed"],
         "splits": cfg["dataset"]["splits"], "preprocess": cfg["preprocess"],
         "per_class": counts.to_dict(orient="records")},
        open(paths["split_report"], "w", encoding="utf-8"), indent=2)

    plot_class_distribution(counts, paths["plots"] / "class_distribution.png")

    print(f"\nReport   : {paths['results'] / 'dataset_report.md'}")
    print(f"Manifest : {paths['manifest']}")
    print("Next     : python -m pytest test_dataset.py   then   python scripts/run_all.py")


if __name__ == "__main__":
    main()
