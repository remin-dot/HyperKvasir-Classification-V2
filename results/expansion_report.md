# V.2 Training-Set Expansion Report

`val/` and `test/` are untouched: 1,598 and 1,610 images hardlinked from
V.1's seed-2024 split. Only `train/` grew. This is what makes the V.1 -> V.2
comparison a paired measurement rather than a separate experiment.

## Per-class training images

| Class | V.1 | V.2 | Added |
|---|---:|---:|---:|
| barretts | 28 | 28 | +0 |
| barretts-short-segment | 37 | 37 | +0 |
| bbps-0-1 | 452 | 452 | +0 |
| bbps-2-3 | 803 | 803 | +0 |
| cecum | 706 | 706 | +0 |
| dyed-lifted-polyps | 701 | 701 | +0 |
| dyed-resection-margins | 692 | 692 | +0 |
| esophagitis-a | 282 | 282 | +0 |
| esophagitis-b-d | 182 | 182 | +0 |
| hemorrhoids | 4 | 4 | +0 |
| ileum | 6 | 6 | +0 |
| impacted-stool | 91 | 91 | +0 |
| polyps | 719 | 1,418 | +699 |
| pylorus | 699 | 699 | +0 |
| retroflex-rectum | 273 | 273 | +0 |
| retroflex-stomach | 534 | 534 | +0 |
| ulcerative-colitis-grade-0-1 | 24 | 24 | +0 |
| ulcerative-colitis-grade-1 | 140 | 140 | +0 |
| ulcerative-colitis-grade-1-2 | 7 | 7 | +0 |
| ulcerative-colitis-grade-2 | 310 | 310 | +0 |
| ulcerative-colitis-grade-2-3 | 19 | 19 | +0 |
| ulcerative-colitis-grade-3 | 93 | 93 | +0 |
| z-line | 652 | 652 | +0 |
| **total** | **7,454** | **8,153** | **+699** |

Class imbalance: **201:1** in V.1 -> **354:1** in V.2.

## Leakage gate

Every candidate image was checked against the md5 and the filename stem of
all 3,208 held-out images before being written.

| Rejected for | Count |
|---|---:|
| filename_stem_in_val_or_test | 301 |
| md5_in_val_or_test | 301 |
| caught_by_both_gates | 301 |

A zero here is not automatically good news -- it can equally mean the gate
never saw the data it was meant to catch. Read it together with the stage
counts above.

## Stages

### segmented

```json
{
  "candidates": 1000,
  "added": 699
}
```

## Known limitation: pseudo-label confirmation bias

The pseudo-labels come from V.1's own model, macro-F1 0.6021. Its errors
become V.2's training targets, and no confidence threshold removes that -- a
confidently wrong prediction is exactly the kind that survives filtering.
The control is that `test/` is frozen and was never pseudo-labelled, so the
reported number is measured against ground truth. This is a real limitation
and is reported whatever direction the result goes.
