"""Dataset integrity checks. Run these before committing hours to training.

    python -m pytest test_dataset.py -v
    python test_dataset.py                 # same checks, no pytest needed

The leakage check is the one that matters: if the same image lands in both train
and test, every metric in the final report is inflated and meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import load_config  # noqa: E402

CFG = load_config()
SPLITS = ("train", "val", "test")


def _require_prepared():
    if not CFG["paths"]["manifest"].exists():
        pytest.skip("dataset not prepared yet — run: python scripts/prepare_dataset.py")


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    _require_prepared()
    return pd.read_csv(CFG["paths"]["manifest"])


def test_manifest_exists():
    _require_prepared()
    assert CFG["paths"]["manifest"].exists()


def test_split_counts_sum_to_total(manifest):
    per_split = manifest["split"].value_counts()
    assert set(per_split.index) == set(SPLITS), f"unexpected splits: {list(per_split.index)}"
    assert per_split.sum() == len(manifest)


def test_no_hash_leakage_between_splits(manifest):
    """No identical image may appear in more than one split."""
    hashes = {s: set(manifest.loc[manifest["split"] == s, "md5"]) for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = hashes[a] & hashes[b]
        assert not overlap, f"{len(overlap)} duplicate image(s) shared between {a} and {b}"


def test_no_duplicate_hashes_at_all(manifest):
    dupes = manifest["md5"].duplicated().sum()
    assert dupes == 0, f"{dupes} duplicate hashes survived deduplication"


def test_every_class_present_in_every_split(manifest):
    classes = set(manifest["class"])
    for split in SPLITS:
        present = set(manifest.loc[manifest["split"] == split, "class"])
        missing = classes - present
        assert not missing, f"classes missing from '{split}': {sorted(missing)}"


def test_class_ordering_identical_across_splits():
    """The comparison assumes index i means the same class for all four models.

    torchvision ImageFolder and Ultralytics both sort directory names, so this
    holds as long as all three split folders contain the same class set.
    """
    _require_prepared()
    orders = {s: sorted(d.name for d in CFG["paths"][s].iterdir() if d.is_dir())
              for s in SPLITS}
    assert orders["train"] == orders["val"] == orders["test"], orders


def test_files_on_disk_match_manifest(manifest):
    _require_prepared()
    for split in SPLITS:
        expected = int((manifest["split"] == split).sum())
        actual = sum(1 for _ in CFG["paths"][split].rglob("*.jpg"))
        assert actual == expected, f"{split}: {actual} files on disk vs {expected} in manifest"


def test_all_images_readable(manifest):
    _require_prepared()
    assert manifest["ok"].all(), f"{(~manifest['ok']).sum()} unreadable image(s) in the manifest"


def test_no_split_is_empty(manifest):
    for split in SPLITS:
        assert (manifest["split"] == split).sum() > 0, f"'{split}' split is empty"


# --------------------------------------------------------------------------
# V.2 leakage gates
#
# V.2 adds ~90k images to train/ from three further archives while val/ and
# test/ stay frozen at V.1's seed-2024 split. Every number V.2 reports is a
# comparison against V.1 on that exact test set, so contamination would not
# produce an obviously broken result -- it would produce a better-looking one.
# These run against the files actually on disk, not against a manifest.
#
# Note on hashing: V.1's manifest md5 is of the RAW source file, while the
# split folders hold RE-ENCODED images. The two are not comparable, so the
# on-disk checks below hash processed-against-processed. PIL's JPEG encoder is
# deterministic, so a leaked image re-encoded with identical settings yields an
# identical file -- which is exactly what these tests look for.
# --------------------------------------------------------------------------

ADDED_PREFIXES = ("pl_", "vid_", "seg_")


def _require_v2_train():
    if not CFG["paths"]["train"].exists() or not any(CFG["paths"]["train"].iterdir()):
        pytest.skip("V.2 train/ not built yet")


def _hash_split(split: str) -> dict[str, str]:
    """md5 -> relative path, for every image in a split folder."""
    from concurrent.futures import ThreadPoolExecutor

    from common.data import _md5

    files = sorted(CFG["paths"][split].rglob("*.jpg"))
    with ThreadPoolExecutor(max_workers=16) as pool:
        digests = list(pool.map(_md5, files))
    return {d: str(f) for d, f in zip(digests, files)}


@pytest.fixture(scope="module")
def heldout_hashes() -> dict[str, str]:
    _require_v2_train()
    out = {}
    for split in ("val", "test"):
        out.update(_hash_split(split))
    return out


def test_v2_heldout_counts_match_v1():
    """val/ and test/ must still be V.1's frozen split, image for image."""
    _require_v2_train()
    assert sum(1 for _ in CFG["paths"]["val"].rglob("*.jpg")) == 1598
    assert sum(1 for _ in CFG["paths"]["test"].rglob("*.jpg")) == 1610


def test_no_added_images_in_heldout_splits():
    """Nothing V.2 generated may appear in val/ or test/ -- pseudo-labels and
    video frames belong to train/ exclusively."""
    _require_v2_train()
    for split in ("val", "test"):
        offenders = [p.name for p in CFG["paths"][split].rglob("*.jpg")
                     if p.name.startswith(ADDED_PREFIXES)]
        assert not offenders, f"{len(offenders)} generated image(s) in {split}: {offenders[:5]}"


def test_no_hash_shared_between_train_and_heldout(heldout_hashes):
    """The gate the whole project rests on: no training image is byte-identical
    to a held-out image after identical preprocessing."""
    train_hashes = _hash_split("train")
    overlap = set(train_hashes) & set(heldout_hashes)
    examples = [(train_hashes[h], heldout_hashes[h]) for h in list(overlap)[:5]]
    assert not overlap, (
        f"{len(overlap)} training image(s) are byte-identical to held-out "
        f"images. Examples: {examples}"
    )


def test_no_source_stem_shared_between_train_and_heldout():
    """HyperKvasir reuses UUID filenames across archives, so a re-encoded copy
    of a held-out image survives the hash check. Compare source stems too."""
    _require_v2_train()

    def source_stem(p: Path) -> str:
        stem = p.stem
        for prefix in ADDED_PREFIXES:
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    heldout = {source_stem(p) for s in ("val", "test")
               for p in CFG["paths"][s].rglob("*.jpg")}
    collisions = sorted({source_stem(p) for p in CFG["paths"]["train"].rglob("*.jpg")}
                        & heldout)
    assert not collisions, (
        f"{len(collisions)} training image(s) share a source filename with a "
        f"held-out image: {collisions[:5]}"
    )


def test_video_frames_never_split_across_sets():
    """Every frame of a given video goes to train/ only, so temporally adjacent
    near-duplicate frames cannot straddle the train/test boundary."""
    _require_v2_train()
    for split in ("val", "test"):
        assert not list(CFG["paths"][split].rglob("vid_*.jpg")), \
            f"video frames found in {split}"


def test_v2_class_set_unchanged():
    """The expansion must not invent a 24th class or drop one -- V.1's metrics
    are 23-class and the comparison assumes the same label space."""
    _require_v2_train()
    orders = {s: sorted(d.name for d in CFG["paths"][s].iterdir() if d.is_dir())
              for s in SPLITS}
    assert orders["train"] == orders["val"] == orders["test"], orders
    assert len(orders["train"]) == 23, f"expected 23 classes, got {len(orders['train'])}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
