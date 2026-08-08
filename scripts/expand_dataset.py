"""Build the V.2 training set from the rest of the HyperKvasir release.

V.1 trained on 7,454 labeled images and concluded the ceiling was data, not
architecture. V.2 adds the other three archives to `train/` ONLY:

    segmented   1,000 polyp images          -> class `polyps` (a real label)
    videos      373 labeled videos          -> frames, label from the video's finding
    unlabeled   99,417 images, no labels    -> pseudo-labelled by V.1's model

`val/` and `test/` are never touched. They are hardlinked from V.1's seed-2024
split and stay byte-identical, which is the only reason the V.1 -> V.2 numbers
can be compared at all.

    python scripts/expand_dataset.py --stage segmented
    python scripts/expand_dataset.py --stage frames
    python scripts/expand_dataset.py --stage pseudo
    python scripts/expand_dataset.py --stage report
    python scripts/expand_dataset.py --stage all

Stages are independent and resumable: each one skips images already written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import class_names, ensure_dirs, load_config, set_seed  # noqa: E402
from common.data import IMAGE_EXTS, _md5, crop_borders, resize_max_side  # noqa: E402


# --------------------------------------------------------------------------
# Leakage gates
# --------------------------------------------------------------------------
class Gate:
    """Rejects any candidate image that could contaminate val/ or test/.

    Adding ~90k images to a project whose entire claim rests on comparing
    against a fixed test set creates several ways to silently inflate the
    result. Two are checkable from V.1's manifest and both are checked:

      md5    -- byte-identical copy of a held-out image.
      stem   -- HyperKvasir reuses the same UUID filename across archives, so a
                re-encoded copy of a held-out image has a different md5 but the
                same stem. The segmentation set is drawn from the labeled polyps
                class, which makes this a live risk, not a theoretical one.

    Rejections are counted per reason and reported. A rejection count of zero is
    itself worth a second look.
    """

    def __init__(self, cfg: dict):
        gates = cfg["expand"]["gates"]
        manifest_path = cfg["paths"]["v1_manifest"]
        if not manifest_path.exists():
            raise SystemExit(
                f"V.1 manifest not found: {manifest_path}\n"
                f"It carries the md5 of every held-out image and without it the "
                f"leakage gates cannot run. Copy it from V.1's data root."
            )
        df = pd.read_csv(manifest_path)
        held = df[df["split"].isin(["val", "test"])]
        self.use_md5 = bool(gates.get("md5", True))
        self.use_stem = bool(gates.get("filename_stem", True))
        self.md5s = set(held["md5"].dropna()) if self.use_md5 else set()
        self.stems = {Path(p).stem for p in held["path"]} if self.use_stem else set()
        # Also block anything already in train/, so reruns do not duplicate work.
        self.seen_md5: set[str] = set()
        self.rejected = Counter()
        print(f"Leakage gate armed: {len(self.md5s):,} held-out hashes, "
              f"{len(self.stems):,} held-out filename stems.")

    def check(self, path: Path, digest: str | None = None) -> bool:
        """True if this candidate is safe to add to train/.

        Both gates are evaluated independently even once one has already
        rejected the image. Short-circuiting would make the report attribute
        every rejection to whichever check happened to run first, which says
        nothing about whether the other one was load-bearing.
        """
        by_stem = self.use_stem and path.stem in self.stems
        by_md5 = False
        dup = False
        if self.use_md5:
            digest = digest or _md5(path)
            by_md5 = digest in self.md5s
            dup = not by_md5 and digest in self.seen_md5
            if not (by_stem or by_md5 or dup):
                self.seen_md5.add(digest)

        if by_stem:
            self.rejected["filename_stem_in_val_or_test"] += 1
        if by_md5:
            self.rejected["md5_in_val_or_test"] += 1
        if by_stem and by_md5:
            self.rejected["caught_by_both_gates"] += 1
        if by_stem and not by_md5:
            self.rejected["stem_only_would_have_been_missed_by_md5"] += 1
        if dup:
            self.rejected["duplicate_within_new_data"] += 1
        return not (by_stem or by_md5 or dup)


def preprocess_to(src: Path, dst: Path, cfg: dict) -> bool:
    """Write src to dst under V.1's exact preprocessing.

    Baked into the file rather than applied in code, for the same reason V.1
    did it: Ultralytics reads these folders directly and never runs a transform
    pipeline, so preprocessing left in code would apply to some images and not
    others.
    """
    pre = cfg["preprocess"]
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if pre["crop_borders"]:
                im, _ = crop_borders(im)
            if pre["resize_max_side"]:
                im = resize_max_side(im, pre["resize_max_side"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=pre["jpeg_quality"])
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {src.name}: {exc}")
        return False


def _added_counts(cfg: dict) -> dict:
    train = cfg["paths"]["train"]
    return {c.name: len(list(c.glob("*.jpg"))) for c in sorted(train.iterdir()) if c.is_dir()}


# --------------------------------------------------------------------------
# Stage: segmented
# --------------------------------------------------------------------------
def stage_segmented(cfg: dict, gate: Gate) -> dict:
    """The 1,000-image polyp segmentation set. Images only -- the masks are for
    a segmentation task this project does not run."""
    src_root = cfg["paths"]["raw_segmented"]
    if not src_root.exists():
        return {"skipped": "raw_segmented not present -- download the segmented component"}

    cls = cfg["expand"]["segmented"]["class_name"]
    dst_dir = cfg["paths"]["train"] / cls
    candidates = [p for p in src_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
                  and "mask" not in p.parts[-2].lower()]
    print(f"\n[segmented] {len(candidates):,} candidate images -> class '{cls}'")

    added = 0
    for p in candidates:
        dst = dst_dir / f"seg_{p.stem}.jpg"
        if dst.exists():
            continue
        if not gate.check(p):
            continue
        added += preprocess_to(p, dst, cfg)
    print(f"[segmented] added {added:,}")
    return {"candidates": len(candidates), "added": added}


# --------------------------------------------------------------------------
# Stage: frames
# --------------------------------------------------------------------------
def _map_class(name: str, classes: list[str], aliases: dict) -> str | None:
    key = name.strip().lower().replace(" ", "-").replace("_", "-")
    if key in classes:
        return key
    return aliases.get(key)


def stage_frames(cfg: dict, gate: Gate) -> dict:
    """Sample frames from the 373 labeled videos.

    The label is the video's finding folder, mapped onto the 23 image classes.
    Anything that will not map is DROPPED AND COUNTED -- guessing a mapping
    would put wrong labels into training data, which is worse than having less
    of it.

    Every frame of a video goes to train/ only. A video is never split across
    sets, so temporally adjacent near-duplicate frames cannot straddle the
    train/test boundary.
    """
    import cv2  # already a dependency via ultralytics + grad-cam

    src_root = cfg["paths"]["raw_videos"]
    if not src_root.exists():
        return {"skipped": "raw_videos not present -- download the videos component"}

    fcfg = cfg["expand"]["frames"]
    classes = class_names(cfg)
    aliases = {k.lower(): v for k, v in (fcfg.get("aliases") or {}).items()}
    interval = float(fcfg["interval_seconds"])
    cap_per_class = int(fcfg["max_per_class"])

    videos = [p for p in src_root.rglob("*") if p.suffix.lower() in {".avi", ".mp4", ".mkv", ".mov"}]
    print(f"\n[frames] {len(videos)} videos, 1 frame / {interval}s, cap {cap_per_class}/class")

    per_class = Counter(_added_counts(cfg))
    baseline = dict(per_class)
    unmapped: Counter = Counter()
    written = 0
    used_videos = 0

    for vid in sorted(videos):
        # The finding is the immediate parent folder; fall back to grandparent.
        label = _map_class(vid.parent.name, classes, aliases) or \
                _map_class(vid.parent.parent.name, classes, aliases)
        if label is None:
            unmapped[vid.parent.name] += 1
            continue

        room = cap_per_class - (per_class[label] - baseline.get(label, 0))
        if room <= 0:
            continue

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"  ! could not open {vid.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * interval)))
        dst_dir = cfg["paths"]["train"] / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        idx = kept = 0
        while room > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                dst = dst_dir / f"vid_{vid.stem}_{idx:07d}.jpg"
                if not dst.exists():
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im = Image.fromarray(rgb)
                    if cfg["preprocess"]["crop_borders"]:
                        im, _ = crop_borders(im)
                    if cfg["preprocess"]["resize_max_side"]:
                        im = resize_max_side(im, cfg["preprocess"]["resize_max_side"])
                    im.save(dst, "JPEG", quality=cfg["preprocess"]["jpeg_quality"])
                    # Gate after writing: a frame has no source file to hash first.
                    if not gate.check(dst):
                        dst.unlink(missing_ok=True)
                    else:
                        kept += 1
                        room -= 1
            idx += 1
        cap.release()
        per_class[label] += kept
        written += kept
        used_videos += 1

    if unmapped:
        print(f"[frames] {sum(unmapped.values())} videos dropped, finding did not map "
              f"to any of the 23 classes: {dict(unmapped.most_common(10))}")
    print(f"[frames] {written:,} frames from {used_videos} videos")
    return {"videos_found": len(videos), "videos_used": used_videos,
            "frames_added": written, "unmapped_findings": dict(unmapped)}


# --------------------------------------------------------------------------
# Stage: pseudo
# --------------------------------------------------------------------------
def stage_pseudo(cfg: dict, gate: Gate) -> dict:
    """Pseudo-label the 99,417 unlabeled images with V.1's YOLO11 checkpoint.

    Only predictions at or above the confidence threshold are kept, and each
    class is capped. The cap is the point: without it the head classes absorb
    almost all of the 99k and the tail -- which is where V.1 actually failed --
    gets nothing.

    This is confirmation-biased by construction: the labels come from a model
    with macro-F1 0.60, so its mistakes become training targets. The honest
    control is that the test set is frozen and was never pseudo-labelled. Stated
    in the report as a limitation regardless of what the number does.
    """
    from ultralytics import YOLO

    src_root = cfg["paths"]["raw_unlabeled"]
    if not src_root.exists():
        return {"skipped": "raw_unlabeled not present -- download the unlabeled component"}

    pcfg = cfg["expand"]["pseudo"]
    ckpt = cfg["paths"]["baseline_checkpoint"]
    if not ckpt.exists():
        raise SystemExit(f"Baseline checkpoint not found: {ckpt}")

    classes = class_names(cfg)
    conf = float(pcfg["confidence"])
    cap = int(pcfg["max_per_class"])
    batch = int(pcfg["batch_size"])

    images = [p for p in src_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    print(f"\n[pseudo] {len(images):,} unlabeled images, conf>={conf}, cap {cap}/class")
    if not images:
        return {"skipped": "no images found under raw_unlabeled"}

    model = YOLO(str(ckpt))
    # Ultralytics' class order comes from the folder names it trained on, which
    # is the same sorted order class_names() returns. Verify rather than assume:
    # a silent mismatch here would mislabel every pseudo-label.
    names = [model.names[i] for i in range(len(model.names))]
    if names != classes:
        raise SystemExit(
            f"Class order mismatch between checkpoint and train/ folders.\n"
            f"  checkpoint: {names}\n  folders   : {classes}"
        )

    kept_per_class: Counter = Counter()
    conf_hist: dict = defaultdict(list)
    accepted = skipped_conf = skipped_cap = 0

    for start in range(0, len(images), batch):
        chunk = images[start:start + batch]
        results = model.predict(chunk, verbose=False, device=0, imgsz=cfg["models"]["yolo11"]["img_size"])
        writes = []
        for src, res in zip(chunk, results):
            probs = res.probs
            score = float(probs.top1conf)
            label = classes[int(probs.top1)]
            if score < conf:
                skipped_conf += 1
                continue
            if kept_per_class[label] >= cap:
                skipped_cap += 1
                continue
            dst = cfg["paths"]["train"] / label / f"pl_{src.stem}.jpg"
            if dst.exists():
                continue
            if not gate.check(src):
                continue
            kept_per_class[label] += 1
            conf_hist[label].append(score)
            writes.append((src, dst))

        with ThreadPoolExecutor(max_workers=12) as pool:
            accepted += sum(pool.map(lambda a: preprocess_to(a[0], a[1], cfg), writes))

        done = start + len(chunk)
        if done % (batch * 20) == 0 or done >= len(images):
            print(f"  {done:,}/{len(images):,}  accepted {accepted:,}  "
                  f"below-conf {skipped_conf:,}  capped {skipped_cap:,}")

    summary = {
        "scanned": len(images), "accepted": accepted,
        "below_confidence": skipped_conf, "hit_class_cap": skipped_cap,
        "per_class": dict(kept_per_class),
        "mean_confidence": {k: round(sum(v) / len(v), 4) for k, v in conf_hist.items()},
    }
    print(f"[pseudo] accepted {accepted:,} of {len(images):,}")
    return summary


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(cfg: dict, stages: dict, gate: Gate) -> Path:
    counts = _added_counts(cfg)
    v1 = pd.read_csv(cfg["paths"]["v1_manifest"])
    v1_train = v1[v1["split"] == "train"]["class"].value_counts().to_dict()

    lines = [
        "# V.2 Training-Set Expansion Report", "",
        "`val/` and `test/` are untouched: 1,598 and 1,610 images hardlinked from",
        "V.1's seed-2024 split. Only `train/` grew. This is what makes the V.1 -> V.2",
        "comparison a paired measurement rather than a separate experiment.", "",
        "## Per-class training images", "",
        "| Class | V.1 | V.2 | Added |", "|---|---:|---:|---:|",
    ]
    total_v1 = total_v2 = 0
    for cls in sorted(counts):
        a, b = v1_train.get(cls, 0), counts[cls]
        total_v1 += a
        total_v2 += b
        lines.append(f"| {cls} | {a:,} | {b:,} | +{b - a:,} |")
    lines += [f"| **total** | **{total_v1:,}** | **{total_v2:,}** | **+{total_v2 - total_v1:,}** |", ""]

    if total_v1:
        imbalance_v1 = max(v1_train.values()) / max(1, min(v1_train.values()))
        imbalance_v2 = max(counts.values()) / max(1, min(counts.values()))
        lines += [
            f"Class imbalance: **{imbalance_v1:.0f}:1** in V.1 -> "
            f"**{imbalance_v2:.0f}:1** in V.2.", "",
        ]

    lines += ["## Leakage gate", "",
              "Every candidate image was checked against the md5 and the filename stem of",
              "all 3,208 held-out images before being written.", "",
              "| Rejected for | Count |", "|---|---:|"]
    if gate.rejected:
        for reason, n in gate.rejected.most_common():
            lines.append(f"| {reason} | {n:,} |")
    else:
        lines.append("| (nothing rejected) | 0 |")
    lines += ["",
              "A zero here is not automatically good news -- it can equally mean the gate",
              "never saw the data it was meant to catch. Read it together with the stage",
              "counts above.", ""]

    lines += ["## Stages", ""]
    for name, data in stages.items():
        lines += [f"### {name}", "", "```json", json.dumps(data, indent=2)[:4000], "```", ""]

    lines += [
        "## Known limitation: pseudo-label confirmation bias", "",
        "The pseudo-labels come from V.1's own model, macro-F1 0.6021. Its errors",
        "become V.2's training targets, and no confidence threshold removes that -- a",
        "confidently wrong prediction is exactly the kind that survives filtering.",
        "The control is that `test/` is frozen and was never pseudo-labelled, so the",
        "reported number is measured against ground truth. This is a real limitation",
        "and is reported whatever direction the result goes.", "",
    ]

    out = cfg["paths"]["results"] / "expansion_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (cfg["paths"]["results"] / "expansion_summary.json").write_text(
        json.dumps({"per_class": counts, "stages": stages,
                    "gate_rejections": dict(gate.rejected)}, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["segmented", "frames", "pseudo", "report", "all"])
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    set_seed(cfg["seed"])

    if not cfg["paths"]["train"].exists():
        raise SystemExit(
            f"{cfg['paths']['train']} does not exist. The V.2 train/ tree starts as a "
            f"hardlinked copy of V.1's train split -- create it before expanding."
        )

    gate = Gate(cfg)
    stages: dict = {}
    run = args.stage

    if run in ("segmented", "all") and cfg["expand"]["segmented"]["enabled"]:
        stages["segmented"] = stage_segmented(cfg, gate)
    if run in ("frames", "all") and cfg["expand"]["frames"]["enabled"]:
        stages["frames"] = stage_frames(cfg, gate)
    if run in ("pseudo", "all") and cfg["expand"]["pseudo"]["enabled"]:
        stages["pseudo"] = stage_pseudo(cfg, gate)

    if run in ("report", "all") or stages:
        prev = cfg["paths"]["results"] / "expansion_summary.json"
        if run == "report" and prev.exists():
            stages = json.loads(prev.read_text(encoding="utf-8")).get("stages", {})
        out = write_report(cfg, stages, gate)
        print(f"\nReport: {out}")

    total = sum(_added_counts(cfg).values())
    print(f"train/ now holds {total:,} images")
    print("Next: python -m pytest test_dataset.py -v   (leakage gates must pass)")


if __name__ == "__main__":
    main()
