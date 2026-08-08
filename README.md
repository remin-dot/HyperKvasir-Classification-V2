# HyperKvasir V.2 — Does More Data Break the Ceiling?

### Training YOLO11s-cls on the full 58.6 GB HyperKvasir release

A direct follow-up to [V.1](#relationship-to-v1), which benchmarked four architectures on
HyperKvasir's **labeled** archive (10,662 images, 23 classes) and ended with a specific,
testable claim:

> **The performance ceiling is data, not architecture.** Three of four models landed within
> 0.02 macro-F1 of each other. Larger backbones or longer training will not help; the binding
> constraint is 7,454 training images with five classes under 30 examples.

V.1 could not test that claim, because it only ever used 6.7% of the dataset. **V.2 tests it.**
It takes V.1's winning model, unchanged, and trains it on the rest of the release — 99,417
unlabeled images, 373 labeled videos, and the 1,000-image segmentation set — measuring the
result on **V.1's frozen test set**.

If macro-F1 moves, the ceiling was data. If it does not, the ceiling is something else, and
that is the more interesting finding.

---

## The one design decision that matters

Everything here depends on a single constraint:

> **`val/` and `test/` are byte-identical to V.1's seed-2024 split and are never regenerated.**
> Only `train/` grows.

1,598 validation and 1,610 test images are hardlinked out of V.1's data root. This makes
V.1 → V.2 a **paired measurement on identical held-out data**, not a new experiment that
happens to produce a nicer number. Every claim in this repo is downstream of that constraint,
which is why it is enforced by tests that run before training rather than by convention.

Adding ~90,000 images creates several new routes to contaminating that test set. Four gates
close them, and all four are hard failures:

| Gate | Catches |
|---|---|
| **md5** | A byte-identical copy of a held-out image arriving from another archive. |
| **filename stem** | A *re-encoded* copy. HyperKvasir reuses UUID filenames across archives — the segmentation set is drawn from the labeled polyps class — so a hash check alone misses it. |
| **video atomicity** | Every frame of a video goes to `train/` only, so 30 fps near-duplicates cannot straddle the boundary. |
| **class-space** | The expansion cannot invent a 24th class or drop one; V.1's metrics are 23-class. |

```bash
python -m pytest test_dataset.py -v
```

**If these fail, training does not start.**

---

## Status

The dataset is 58.6 GB and the measured link speed is **~240–390 KB/s** (verified against
three hosts, including a CDN — it is the local connection, not the dataset host). The two
large archives therefore take **26–46 hours** to transfer.

| Stage | State |
|---|---|
| Pipeline, config, leakage gates | ✅ complete, gates passing on the base tree |
| `segmented` archive (46 MB) | ✅ downloaded, 1,000 images + masks |
| `unlabeled` archive (29.4 GB) | ⏳ downloading |
| `videos` archive (25.2 GB) | ⏸ queued |
| Expansion, training, evaluation | ⏸ blocked on the transfers |

Results tables are absent rather than provisional. Nothing is reported here until it has been
measured — the same rule V.1 held to.

---

## How the extra data is used

Only the labeled archive ships clean 23-class labels. The other three need handling, and each
approach has a cost that is stated rather than buried.

| Source | Images | Label comes from | Cost |
|---|---:|---|---|
| Labeled (V.1) | 7,454 | Ground truth | — |
| Segmentation set | ~1,000 | Ground truth (all polyps) | Overlaps the labeled polyps class; the stem gate removes held-out copies. |
| Labeled videos | ~20,000 frames | The video's finding, mapped to the 23 classes | Video-level label applied to every frame. Unmappable findings are **dropped and counted**, never guessed. |
| Unlabeled | ~50,000–65,000 | **Pseudo-labelled by V.1's model** | Confirmation bias — see below. |

### The limitation this project cannot engineer away

Pseudo-labels come from V.1's own model, macro-F1 **0.6021**. Its mistakes become V.2's
training targets, and no confidence threshold removes that — a *confidently wrong* prediction
is precisely the kind that survives filtering. The threshold (0.90) and the per-class cap
(3,000) limit the damage and stop the head classes from swamping the tail, but they do not
eliminate the effect.

The control is that `test/` is frozen and was never pseudo-labelled, so the reported number is
measured against ground truth. **This is reported as a limitation whichever direction the
result moves.**

### Why not all 1,059,519 video frames

At ~700 img/s that is 25 min/epoch — 12.6 h for 30 epochs — for frames that are near-duplicates
at 30 fps. Sampling one frame every 2 s costs almost no information and saves ~11 hours.
Configurable in `config.yaml`.

---

## Success criterion

Macro-F1 going up is *not* sufficient. V.1's failure was specific and localized:

> **Five classes scored 0.000 F1** — `barretts`, `ulcerative-colitis-grade-2-3`, `ileum`,
> `ulcerative-colitis-grade-1-2`, `hemorrhoids`. Accuracy 0.9006 against macro-F1 0.6021 is
> the entire story of this dataset.

So the result is judged on **those five classes moving off zero**. A macro-F1 rise driven only
by classes that already worked would mean the extra data did nothing that mattered, and the
report will say so.

| Baseline to beat (V.1, YOLO11s-cls, same 1,610 test images) | |
|---|---:|
| Macro-F1 | 0.6021 |
| Accuracy | 0.9006 |
| Macro ROC-AUC | 0.9875 |
| 3-seed spread (the noise floor) | ± 0.0170 |

A gain smaller than ±0.0170 is **not a result**. V.1's central finding was that a single-split
ranking is not defensible; V.2 does not get to forget that because the number moved the right way.

---

## Running it

Requires Python 3.10+, an NVIDIA GPU with ≥6 GB VRAM, and ~125 GB free disk (peak; ~55 GB is
reclaimable after extraction).

Install PyTorch **first and separately**, or pip resolves the CPU-only build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```bash
pip install -r requirements.txt
```

Fetch the archives. Each is resumable — an interrupted transfer continues rather than
restarting, which matters over 26+ hours:

```bash
python scripts/download_dataset.py --component unlabeled
```

```bash
python scripts/download_dataset.py --component videos
```

```bash
python scripts/download_dataset.py --component segmented
```

Build the expanded training set (stages are independent and skip completed work):

```bash
python scripts/expand_dataset.py --stage all
```

Verify no contamination, then train:

```bash
python -m pytest test_dataset.py -v
```

```bash
python scripts/run_all.py --only yolo11
```

⚠️ Do **not** run `scripts/prepare_dataset.py` in V.2 — it calls `materialize()`, which wipes
and rebuilds all three split folders. That would destroy the frozen val/test split and silently
void the comparison. It is retained only because `run_all.py` references it for the V.1 path.

---

## Project layout

```
config.yaml                  all tunables, incl. the expand: block
baseline_v1/                 V.1's checkpoint, metrics and 3-seed spread
  yolo11_best.pt             the pseudo-labeller AND the number to beat
common/                      shared implementation, inherited from V.1 unchanged
scripts/
  download_dataset.py        + --component {labeled,unlabeled,segmented,videos}
  expand_dataset.py          NEW: segmented / frames / pseudo -> train/
  run_all.py, evaluate_all.py, generate_gradcam.py, ...
test_dataset.py              V.1's 9 integrity checks + V.2's 6 leakage gates
```

Data lives at `D:/HyperKvasir_v2`, off OneDrive — 100k files inside a syncing folder makes
every epoch I/O-bound. Override with `HK_DATA_ROOT`.

---

## Relationship to V.1

V.1 is a separate, complete project: a four-model benchmark (YOLOv8s-cls, YOLO11s-cls,
ResNet-50, EfficientNetV2-S) with Grad-CAM explainability and 3-seed validation. Its central
finding was that the top three models are **statistically indistinguishable** — the gap
between them (0.0120) is smaller than the run-to-run spread (0.0130) — and that YOLO11s-cls is
the right pick not because it is more accurate but because it matches the others at 4× fewer
parameters and 3.5× faster inference.

V.2 inherits its pipeline, its metric code, its preprocessing and its test set. It changes one
variable: how much data goes in.

---

## License and citation

Code: MIT. Dataset: **CC BY 4.0**, not redistributed here.

> Borgli, H., Thambawita, V., Smedsrud, P.H. et al. *HyperKvasir, a comprehensive multi-class
> image and video dataset for gastrointestinal endoscopy.* Scientific Data **7**, 283 (2020).
> https://doi.org/10.1038/s41597-020-00622-y

**Research and educational use only. Not validated for, and not to be used for, clinical
diagnosis.**
