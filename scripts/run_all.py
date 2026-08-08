"""Run the complete pipeline end to end.

    python scripts/run_all.py                        # everything
    python scripts/run_all.py --only resnet --epochs 2   # 5-minute smoke test
    python scripts/run_all.py --stage evaluate compare   # selected stages
    python scripts/run_all.py --skip download            # data already in place

Stages are resumable: each one skips itself if its outputs already exist, unless
--force is given. A model that fails does not abort the run -- the failure is
recorded and reported.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ensure_dirs, load_config  # noqa: E402

STAGES = ["hardware", "download", "prepare", "train", "evaluate", "gradcam",
          "compare", "select", "report"]


def banner(text: str) -> None:
    print(f"\n\n{'#' * 70}\n#  {text}\n{'#' * 70}")


def run_stage(name: str, cfg: dict, args) -> tuple[bool, str]:
    """Each stage runs in-process so peak-VRAM stats and timings stay accurate."""
    only = args.only
    epochs = args.epochs

    if name == "hardware":
        from common import hardware
        hardware.write_report(cfg)
        return True, "ok"

    if name == "download":
        raw = cfg["paths"]["raw"]
        existing = sum(1 for _ in raw.rglob("*.jpg")) if raw.exists() else 0
        if existing > 1000 and not args.force:
            return True, f"skipped ({existing:,} images already present)"
        cmd = [sys.executable, str(Path(__file__).parent / "download_dataset.py")]
        if args.force:
            cmd.append("--force")
        return subprocess.run(cmd).returncode == 0, "ok"

    if name == "prepare":
        if cfg["paths"]["train"].exists() and any(cfg["paths"]["train"].iterdir()) and not args.force:
            return True, "skipped (splits exist)"
        cmd = [sys.executable, str(Path(__file__).parent / "prepare_dataset.py")]
        if args.force:
            cmd.append("--force")
        return subprocess.run(cmd).returncode == 0, "ok"

    if name == "train":
        from scripts.train_all import train_one
        keys = only or list(cfg["models"].keys())
        trained, failed, skipped = [], [], []
        for key in keys:
            if (cfg["paths"]["checkpoints"] / key / "best.pt").exists() and not args.force:
                print(f"[skip] {key}: checkpoint exists")
                skipped.append(key)
                continue
            ok, _ = train_one(key, cfg, epochs)
            (trained if ok else failed).append(key)
        return True, f"trained={trained} skipped={skipped} failed={failed}"

    if name == "evaluate":
        from common import hardware
        from common.evaluate import evaluate_model
        device = hardware.setup_torch(cfg["seed"])
        keys = only or list(cfg["models"].keys())
        done, failed = [], []
        for key in keys:
            if not (cfg["paths"]["checkpoints"] / key / "best.pt").exists():
                print(f"[skip] {key}: no checkpoint")
                continue
            if (cfg["paths"]["results"] / key / "metrics.json").exists() and not args.force:
                print(f"[skip] {key}: metrics.json exists")
                done.append(key)
                continue
            try:
                evaluate_model(key, cfg, device)
                done.append(key)
            except Exception as exc:  # noqa: BLE001
                _record_failure(cfg, key, "evaluation", exc)
                failed.append(key)
        return bool(done), f"evaluated={done} failed={failed}"

    if name == "gradcam":
        from common import hardware
        from common.cam import run_gradcam
        device = hardware.setup_torch(cfg["seed"])
        keys = only or list(cfg["models"].keys())
        done, failed = [], []
        for key in keys:
            if not (cfg["paths"]["results"] / key / "predictions.csv").exists():
                continue
            try:
                run_gradcam(key, cfg, device=device)
                done.append(key)
            except Exception as exc:  # noqa: BLE001
                _record_failure(cfg, key, "gradcam", exc, fatal=False)
                failed.append(key)
        return True, f"gradcam={done} failed={failed}"

    if name in ("compare", "select", "report"):
        script = {"compare": "compare_models.py", "select": "select_best.py",
                  "report": "generate_report.py"}[name]
        return subprocess.run(
            [sys.executable, str(Path(__file__).parent / script)]).returncode == 0, "ok"

    raise ValueError(f"unknown stage {name}")


def _record_failure(cfg: dict, key: str, stage: str, exc: Exception, fatal: bool = True) -> None:
    import json
    tb = traceback.format_exc()
    target = cfg["paths"]["results"] / key if fatal else cfg["paths"]["gradcam"] / key
    target.mkdir(parents=True, exist_ok=True)
    (target / ("FAILED.json" if fatal else "gradcam_info.json")).write_text(json.dumps({
        "model_key": key, "model": key, "stage": stage, "method": None, "images": 0,
        "error_type": type(exc).__name__, "error": str(exc), "traceback": tb,
    }, indent=2), encoding="utf-8")
    (cfg["paths"]["logs"] / f"{key}_{stage}_error.log").write_text(tb, encoding="utf-8")
    print(f"\n{stage.upper()} FAILED for {key}: {type(exc).__name__}: {exc}\n"
          f"Recorded; continuing.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[], help="skip these stages")
    ap.add_argument("--only", nargs="+", metavar="MODEL", help="restrict to these model keys")
    ap.add_argument("--epochs", type=int, help="override config.yaml epochs")
    ap.add_argument("--force", action="store_true", help="redo stages even if outputs exist")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    if args.only:
        unknown = [k for k in args.only if k not in cfg["models"]]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Known: {list(cfg['models'])}")

    stages = [s for s in (args.stage or STAGES) if s not in args.skip]
    print(f"Pipeline : {' -> '.join(stages)}")
    print(f"Models   : {args.only or list(cfg['models'].keys())}")
    print(f"Data root: {cfg['paths']['data_root']}")
    if args.epochs:
        print(f"Epochs   : {args.epochs} (overriding config.yaml)")

    results, start_all = [], time.perf_counter()
    for stage in stages:
        banner(f"STAGE: {stage}")
        t0 = time.perf_counter()
        try:
            ok, note = run_stage(stage, cfg, args)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            (cfg["paths"]["logs"] / f"stage_{stage}_error.log").write_text(tb, encoding="utf-8")
            print(tb)
            ok, note = False, f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        results.append((stage, ok, note, elapsed))
        print(f"\n[{stage}] {'OK' if ok else 'FAILED'} in {elapsed / 60:.1f} min — {note}")

        # Downstream stages are meaningless without data, so stop there only.
        if not ok and stage in ("download", "prepare"):
            print(f"\nStage '{stage}' is a hard prerequisite. Stopping.")
            break

    total = time.perf_counter() - start_all
    banner("PIPELINE SUMMARY")
    print(f"{'stage':<12}{'status':<10}{'minutes':>9}   note")
    for stage, ok, note, elapsed in results:
        print(f"{stage:<12}{'OK' if ok else 'FAILED':<10}{elapsed / 60:>9.1f}   {note}")
    print(f"\nTotal: {total / 60:.1f} min")

    report = cfg["paths"]["results"] / "FINAL_REPORT.md"
    best = cfg["paths"]["results"] / "best_model" / "best_model.txt"
    if best.exists():
        print("\n" + "=" * 70)
        print(best.read_text(encoding="utf-8").split("-" * 70)[0].strip())
        print("=" * 70)
    if report.exists():
        print(f"\nFull report: {report}")


if __name__ == "__main__":
    main()
