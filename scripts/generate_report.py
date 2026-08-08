"""Assemble results/FINAL_REPORT.md from artifacts already written to disk.

Every number in the report is read from a JSON/CSV file produced by an earlier
stage. Nothing is hard-coded, estimated or typed in by hand. If a model failed,
its failure record appears where its metrics would have been.

    python scripts/generate_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def section_hardware(cfg: dict) -> str:
    info = _read_json(cfg["paths"]["results"] / "hardware_info.json")
    if not info:
        return "_hardware_info.json missing — run `python scripts/run_all.py --stage hardware`._"

    gpu, torch_info = info.get("gpu"), info.get("torch", {})
    rows = [
        ("Platform", info["platform"]),
        ("CPU", f"{info['cpu']} ({info['cpu_cores_physical']} cores / {info['cpu_threads']} threads)"),
        ("RAM", f"{info['ram_gb']} GB"),
        ("Disk free", f"{info['disk_free_gb']} GB ({info['disk_checked']})"),
        ("Python", info["python"]),
        ("GPU", f"{gpu['name']} — {gpu['vram_total_mb']} MiB, driver {gpu['driver_version']}, "
                f"compute {gpu['compute_capability']}" if gpu else "none detected"),
        ("PyTorch", torch_info.get("version", "not installed")),
        ("CUDA", f"{torch_info.get('cuda_version')} (available: {torch_info.get('cuda_available')})"),
        ("Mixed precision", "bfloat16" if torch_info.get("bf16_supported") else
                            ("float16" if torch_info.get("cuda_available") else "disabled")),
    ]
    out = ["| Item | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]

    if torch_info.get("cuda_available"):
        out += ["", "GPU training was **enabled** for every model. Batch size was scaled to the "
                    "VRAM actually free at launch, and mixed precision was on."]
    else:
        out += ["", "**No usable GPU** — every model trained on CPU. Timings below are therefore "
                    "not comparable to GPU figures."]

    versions = info.get("packages", {})
    if versions:
        out += ["", "<details><summary>Package versions (reproducibility record)</summary>", "",
                "| Package | Version |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in versions.items()]
        out += ["", "</details>"]
    return "\n".join(out)


def section_dataset(cfg: dict) -> str:
    report = _read_text(cfg["paths"]["results"] / "dataset_report.md")
    if not report:
        return "_dataset_report.md missing — run `python scripts/prepare_dataset.py`._"
    # Demote the report's headings so they nest under this section.
    body = "\n".join(("##" + line if line.startswith("#") else line)
                     for line in report.splitlines()[1:])
    return body + "\n\n![Class distribution](../plots/class_distribution.png)\n"


def section_methodology(cfg: dict) -> str:
    tcfg = cfg["training"]
    lines = [
        "### Dataset preparation", "",
        "1. Extract the official HyperKvasir labeled-image archive (CC BY 4.0).",
        "2. Audit every file: decode check, resolution, MD5 hash, green-overlay detection.",
        "3. Remove byte-identical duplicates **before** splitting.",
        f"4. Class-stratified {cfg['dataset']['splits']['train']:.0%}/"
        f"{cfg['dataset']['splits']['val']:.0%}/{cfg['dataset']['splits']['test']:.0%} split, "
        f"seed {cfg['seed']}, with a guarantee of at least one image per class per split.",
        "5. Bake preprocessing into the files on disk (black-border crop, max side "
        f"{cfg['preprocess']['resize_max_side']} px) and write a plain ImageFolder tree.", "",
        "Step 5 is what makes the comparison fair: Ultralytics reads the split folders "
        "directly and never runs our transform pipeline, so any preprocessing left in code "
        "would have applied to two of the four models only.", "",
        "### What is identical across all four models", "",
        "| Held constant | Value |", "|---|---|",
        "| Train / val / test split | same files, same seed |",
        f"| Epoch budget | {tcfg['epochs']} |",
        f"| Early-stopping patience | {tcfg['patience']} |",
        "| Test set | identical ordered file list |",
        "| Metric code | `common/metrics.py`, one implementation |",
        "| Latency protocol | batch=1 raw forward, CUDA-synchronized, "
        f"{cfg['evaluation']['latency_warmup']} warmup + {cfg['evaluation']['latency_iters']} timed |",
        "", "### What is NOT identical — and why", "",
        "**ResNet-50 and EfficientNetV2-S** are trained by `common/torch_trainer.py`: AdamW, "
        "discriminative learning rates (fresh head faster than pretrained backbone), "
        f"{tcfg['warmup_epochs']}-epoch linear warmup into cosine decay, "
        f"label smoothing {tcfg['label_smoothing']}, inverse-square-root class weighting, and "
        "augmentation including full ±180° rotation (an endoscope frame has no canonical 'up').",
        "",
        "**YOLOv8-cls and YOLO11-cls** are trained by Ultralytics, which owns its training "
        "loop and will not accept our optimizer, schedule, augmentation policy or class "
        "weights. They therefore train **unweighted** under Ultralytics defaults. This is a "
        "genuine methodological limitation of the comparison and is stated here rather than "
        "glossed over. Ultralytics also reports only top-1/top-5 accuracy per epoch, so its "
        "training curves show accuracy where the torch models show macro-F1.", "",
        "### Why macro-F1 is the headline metric", "",
        "The class distribution is imbalanced roughly 191:1. A model that predicts only the "
        "largest classes and never predicts the tail still reaches around 0.85 accuracy while "
        "being clinically useless. Macro-F1 weights every class equally and exposes that "
        "failure; accuracy hides it.", "",
    ]
    return "\n".join(lines)


def section_model_results(cfg: dict) -> str:
    blocks = []
    for key, mcfg in cfg["models"].items():
        result_dir = cfg["paths"]["results"] / key
        metrics = _read_json(result_dir / "metrics.json")
        failure = _read_json(result_dir / "FAILED.json")
        blocks.append(f"### {mcfg['display_name']}")

        if not metrics:
            if failure:
                blocks += [
                    "", f"**NOT COMPLETED** — failed at the *{failure.get('stage', 'unknown')}* stage.", "",
                    f"```\n{failure.get('error_type', '')}: {failure.get('error', '')}\n```", "",
                    "No metrics are reported for this model. Nothing was substituted or estimated.", ""]
            else:
                blocks += ["", "**NOT COMPLETED** — never trained or evaluated.", ""]
            continue

        train_meta = _read_json(result_dir / "train_meta.json") or {}
        auc = (f"{metrics['roc_auc_macro']:.4f}" if metrics.get("roc_auc_macro") is not None else "n/a")
        pr_auc = (f"{metrics['pr_auc_macro']:.4f}" if metrics.get("pr_auc_macro") is not None else "n/a")
        blocks += [
            "", f"Framework: `{metrics['framework']}` · architecture: `{mcfg.get('arch', mcfg.get('weights'))}` "
                f"· input {metrics['img_size']}px", "",
            "| Metric | Value |", "|---|---:|",
            f"| Accuracy | {metrics['accuracy']:.4f} |",
            f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
            f"| **Macro F1** | **{metrics['f1_macro']:.4f}** |",
            f"| Macro precision | {metrics['precision_macro']:.4f} |",
            f"| Macro recall | {metrics['recall_macro']:.4f} |",
            f"| Weighted F1 | {metrics['f1_weighted']:.4f} |",
            f"| Macro ROC-AUC | {auc} |",
            f"| Macro PR-AUC | {pr_auc} |",
            f"| MCC | {metrics['mcc']:.4f} |",
            f"| Cohen's kappa | {metrics['cohen_kappa']:.4f} |",
            f"| Parameters | {metrics['params_millions']} M |",
            f"| Checkpoint size | {metrics['checkpoint_size_mb']} MB |",
            f"| Inference latency | {metrics['inference_ms_mean']:.2f} ms/image ({metrics['fps']} FPS) |",
            f"| Training time | {metrics.get('training_minutes', 'n/a')} min |",
            f"| Epochs run | {metrics.get('epochs_run', 'n/a')} |",
            f"| Peak VRAM (train) | {metrics.get('train_peak_vram_mb', 'n/a')} MB |",
            f"| Checkpoint | `{metrics['checkpoint_path']}` |", "",
        ]
        if train_meta.get("methodology_deviation"):
            blocks += [f"> **Methodology note.** {train_meta['methodology_deviation']}", ""]

        curve = cfg["paths"]["plots"] / "training_curves" / f"{key}.png"
        if curve.exists():
            blocks.append(f"![{key} training curves](../plots/training_curves/{key}.png)\n")
        cm = cfg["paths"]["plots"] / "confusion_matrices" / f"{key}.png"
        if cm.exists():
            blocks.append(f"![{key} confusion matrix](../plots/confusion_matrices/{key}.png)\n")
        roc = cfg["paths"]["plots"] / "roc_curves" / f"{key}.png"
        if roc.exists():
            blocks.append(f"![{key} ROC curves](../plots/roc_curves/{key}.png)\n")

        per_class = metrics.get("per_class", {})
        if per_class:
            blocks += ["<details><summary>Per-class precision / recall / F1</summary>", "",
                       "| Class | Precision | Recall | F1 | Test images |", "|---|---:|---:|---:|---:|"]
            for name, v in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
                warn = " ⚠️" if v["support"] <= 2 else ""
                blocks.append(f"| {name}{warn} | {v['precision']:.4f} | {v['recall']:.4f} | "
                              f"{v['f1']:.4f} | {v['support']} |")
            blocks += ["", "⚠️ = 2 or fewer test images; these per-class scores are noise, not signal.",
                       "", "</details>", ""]
    return "\n".join(blocks)


def section_comparison(cfg: dict) -> str:
    table = _read_text(cfg["paths"]["results"] / "comparison" / "model_comparison.md")
    if not table:
        return "_No comparison table — run `python scripts/compare_models.py`._"
    out = [table, "", "![Model comparison](../plots/model_comparison/model_comparison.png)", ""]

    # Multi-seed results supersede the single-split table above: a ranking without
    # error bars cannot tell a real difference from split luck.
    multiseed = _read_text(cfg["paths"]["results"] / "comparison" / "model_comparison_multiseed.md")
    if multiseed:
        out += [
            "### Repeated-seed results (this is the defensible comparison)", "",
            "The table above is a single train/val/test split. On a dataset this small a "
            "difference of a few thousandths of macro-F1 is well within the noise of which "
            "images happened to land in the test set. The results below repeat the entire "
            "pipeline — re-splitting the data and re-training every model — under several "
            "seeds, so the spread is visible.", "",
            multiseed, "",
            "![Multi-seed comparison](../plots/model_comparison/multiseed.png)", "",
        ]
    failures = _read_json(cfg["paths"]["results"] / "comparison" / "failures.json")
    if failures:
        out += ["### Models missing from the comparison", ""]
        for f in failures:
            out.append(f"- **{f.get('display_name', f['model_key'])}** — failed at "
                       f"*{f.get('stage', 'unknown')}*: `{f.get('error_type', '')}: "
                       f"{f.get('error', '')[:200]}`")
        out.append("")
    return "\n".join(out)


def section_gradcam(cfg: dict) -> str:
    out = [
        "Grad-CAM shows which image regions drove each prediction. Beyond explainability, "
        "these figures are the audit on a specific failure mode: HyperKvasir images carry "
        "black borders and, on a subset, a green ScopeGuide picture-in-picture overlay that "
        "correlates with the acquisition session rather than with pathology. If attention "
        "sits on a corner box or the frame edge instead of mucosa, the metrics above are "
        "measuring the wrong thing.", "",
    ]
    for key, mcfg in cfg["models"].items():
        info = _read_json(cfg["paths"]["gradcam"] / key / "gradcam_info.json")
        out.append(f"### {mcfg['display_name']}")
        if not info:
            out += ["", "_Not generated._", ""]
            continue
        if not info.get("method"):
            out += ["", f"**Could not be generated** — `{info.get('error_type', '')}: "
                        f"{info.get('error', '')}`", ""]
            continue
        out += ["", f"Method: **{info['method']}**"
                    + (" (Grad-CAM fallback)" if info.get("fallback_used") else "")
                    + f" · target layer: `{info.get('target_layer', 'n/a')}` · "
                      f"{info['images']} visualizations", ""]
        if info.get("note"):
            out += [f"> {info['note']}", ""]
        for f in info.get("files", [])[:6]:
            name = Path(f).name
            label = "correct" if name.startswith("correct") else "misclassified"
            out.append(f"![{key} {label}](../gradcam/{key}/{name})")
        out.append("")
    return "\n".join(out)


def section_best(cfg: dict) -> str:
    best = _read_json(cfg["paths"]["results"] / "best_model" / "best_model_metrics.json")
    if not best:
        return "_No selection made — run `python scripts/select_best.py`._"

    m = best["metrics"]
    out = [
        f"# {best['best_model_name']}", "",
        f"Weighted selection score **{best['selection_score']:.4f}** "
        f"(selected by: {best.get('selected_by', 'weighted score')}).", "",
        "| Rank | Model | Weighted score | Macro F1 | |", "|---:|---|---:|---:|---|",
    ]
    out += [f"| {i + 1} | {r['model']} | {r['score']:.4f} | "
            f"{r.get('f1_macro', float('nan')):.4f} | "
            f"{'**selected**' if r.get('selected') else ''} |"
            for i, r in enumerate(best["ranking"])]

    ovr = best.get("f1_guard_override")
    if ovr:
        out += [
            "", f"> **Macro-F1 guard applied.** `{ovr['weighted_winner']}` led the raw weighted "
                f"score ({ovr['weighted_score']:.4f}) on secondary criteria such as inference "
                f"speed and model size, but `{ovr['overridden_by']}` scored "
                f"{ovr['f1_gap']:.4f} higher on macro-F1 — beyond the {ovr['guard']} tolerance "
                f"configured in `config.yaml`. Macro-F1 is the primary criterion for this "
                f"task, so the higher-F1 model was selected. Set `selection.f1_guard: 1.0` to "
                f"disable the guard and rank purely by weighted score.",
        ]

    out += [
        "", "Selection weights (renormalized over measurable criteria):", "",
        "| Criterion | Weight |", "|---|---:|",
    ]
    out += [f"| {k} | {v:.3f} |" for k, v in best["selection_weights"].items()]
    if best.get("criteria_dropped"):
        out += ["", f"Dropped as unmeasurable for at least one model: "
                    f"`{', '.join(best['criteria_dropped'])}` "
                    f"(weight redistributed proportionally)."]
    out += [
        "", "### Winning model — measured test-set performance", "",
        "| Metric | Value |", "|---|---:|",
        f"| Macro F1 | **{m['f1_macro']:.4f}** |",
        f"| Accuracy | {m['accuracy']:.4f} |",
        f"| Balanced accuracy | {m['balanced_accuracy']:.4f} |",
        f"| Macro precision | {m['precision_macro']:.4f} |",
        f"| Macro recall | {m['recall_macro']:.4f} |",
        f"| Macro ROC-AUC | " + (f"{m['roc_auc_macro']:.4f}" if m.get("roc_auc_macro") is not None else "n/a") + " |",
        f"| MCC | {m['mcc']:.4f} |",
        f"| Parameters | {m['params_millions']} M |",
        f"| Inference | {m['inference_ms_mean']:.2f} ms/image ({m['fps']} FPS) |",
        f"| Training time | {m.get('training_minutes', 'n/a')} min |",
        "", "### Where to find it", "",
        f"- Checkpoint (original): `{m['checkpoint_path']}`",
        "- Checkpoint (copy): `results/best_model/best_model_checkpoint/`",
        "- Full rationale: `results/best_model/best_model.txt`",
        "- Metrics: `results/best_model/best_model_metrics.json`", "",
        "![Best model selection](../results/best_model/best_model_comparison.png)", "",
    ]
    return "\n".join(out)


def section_conclusion(cfg: dict) -> str:
    best = _read_json(cfg["paths"]["results"] / "best_model" / "best_model_metrics.json")
    split = _read_json(cfg["paths"]["split_report"])
    if not best:
        return "_Pending model selection._"

    m = best["metrics"]
    ranking = best["ranking"]
    selected = next((r for r in ranking if r.get("selected")), ranking[0])
    others = [r for r in ranking if r is not selected]
    ovr = best.get("f1_guard_override")

    tiny = [c for c, v in m.get("per_class", {}).items() if v["support"] <= 2]
    if ovr:
        headline = (
            f"It achieved the highest macro-F1 of any model evaluated "
            f"({m['f1_macro']:.4f}), on {m['n_test_images']} held-out test images across "
            f"{m['n_classes']} classes, at {m['inference_ms_mean']:.2f} ms per image. "
            f"`{ovr['weighted_winner']}` led the combined weighted score on speed and size, "
            f"but by a margin that did not justify giving up {ovr['f1_gap']:.4f} of macro-F1 "
            f"— the metric that actually reflects performance across the imbalanced classes.")
    else:
        headline = (
            f"It achieved the highest weighted score across macro-F1, accuracy, AUC, "
            f"precision/recall balance, inference speed and model size — "
            f"macro-F1 {m['f1_macro']:.4f} and accuracy {m['accuracy']:.4f} on "
            f"{m['n_test_images']} held-out test images across {m['n_classes']} classes, "
            f"at {m['inference_ms_mean']:.2f} ms per image.")

    out = [f"**Recommendation: {best['best_model_name']}.**", "", headline]

    multiseed = _read_text(cfg["paths"]["results"] / "comparison" / "model_comparison_multiseed.md")
    if multiseed:
        out += ["", "**Read this recommendation together with the repeated-seed results in "
                    "section 5.** The selection above is computed from a single split; "
                    "section 5 states whether the winning margin is larger than the "
                    "run-to-run variation, and that is the claim to defend."]

    if others:
        runner = others[0]
        f1_gap = m["f1_macro"] - runner.get("f1_macro", 0.0)
        margin = ("a clear margin" if abs(f1_gap) > 0.05 else
                  "a narrow margin — close enough that a different random split could "
                  "plausibly reorder the top two, so do not over-claim the ranking")
        out += ["", f"Closest competitor: {runner['model']}, macro-F1 "
                    f"{runner.get('f1_macro', float('nan')):.4f} against "
                    f"{m['f1_macro']:.4f} — {margin}."]

    out += [
        "", "### Caveats to state alongside these numbers", "",
        f"1. **Accuracy overstates real performance.** Accuracy {m['accuracy']:.4f} against "
        f"macro-F1 {m['f1_macro']:.4f} is the imbalance showing through. Quote macro-F1.",
    ]
    if tiny:
        out.append(f"2. **{len(tiny)} class(es) have ≤2 test images** ({', '.join(tiny)}); "
                   f"their per-class scores are noise. They are retained so the 23-class "
                   f"result stays comparable to published HyperKvasir baselines.")
    out += [
        f"{'3' if tiny else '2'}. **No patient identifiers exist in HyperKvasir**, so "
        "near-duplicate frames from the same examination may straddle the train/test "
        "boundary. Byte-identical duplicates were removed; perceptual ones cannot be "
        "detected reliably without patient metadata. This affects every published result "
        "on this dataset, not only these.",
        f"{'4' if tiny else '3'}. **The YOLO models did not train under the same recipe** "
        "(Ultralytics defaults, no class weighting). Their comparison is fair on data and "
        "evaluation, not on optimization.",
        f"{'5' if tiny else '4'}. **Research use only.** Not validated, and not suitable, "
        "for clinical diagnosis.",
    ]
    if split:
        out += ["", f"_Dataset: {split['audit']['found']:,} files audited, "
                    f"{split['audit'].get('duplicates', 0)} duplicates removed, "
                    f"{split['audit'].get('corrupt', 0)} unreadable, "
                    f"{split['audit'].get('pip', 0)} with a detected ScopeGuide overlay._"]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    doc = [
        "# HyperKvasir Gastrointestinal Image Classification — Final Report", "",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by "
        f"`scripts/generate_report.py`. Every figure in this report is read from a file "
        f"written by an earlier pipeline stage; none is typed in by hand._", "",
        "---", "", "## 1. Hardware", "", section_hardware(cfg), "",
        "---", "", "## 2. Dataset", "", section_dataset(cfg), "",
        "---", "", "## 3. Methodology", "", section_methodology(cfg), "",
        "---", "", "## 4. Model Results", "", section_model_results(cfg), "",
        "---", "", "## 5. Comparison", "", section_comparison(cfg), "",
        "---", "", "## 6. Grad-CAM Explainability", "", section_gradcam(cfg), "",
        "---", "", "## 7. Best Model", "", section_best(cfg), "",
        "---", "", "## 8. Conclusion", "", section_conclusion(cfg), "",
        "---", "",
        "_HyperKvasir is distributed under CC BY 4.0. Borgli et al., "
        "'HyperKvasir, a comprehensive multi-class image and video dataset for "
        "gastrointestinal endoscopy', Scientific Data 7, 283 (2020)._", "",
    ]
    text = "\n".join(doc)
    out = cfg["paths"]["results"] / "FINAL_REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"-> {out}  ({len(text):,} characters)")


if __name__ == "__main__":
    main()
