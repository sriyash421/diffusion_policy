"""Reserve 50 of BLQ137's training-region episodes for an independent ID test."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from diffusion_policy.dataset.pusht_image_dataset import load_split_manifest, split_checksum

SOURCE = REPO / "diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json"
DEST = SOURCE.with_name("pusht_blockquad_bottomleft_train87_idtest50_seed42.json")


def holdout_checksum(manifest):
    return hashlib.sha256(json.dumps({k: manifest[k] for k in
        ("train", "val", "test", "id_test")}, sort_keys=True).encode()).hexdigest()


def validate_holdout(manifest, source):
    assert [len(manifest[k]) for k in ("train", "val", "test", "id_test")] == [87, 19, 50, 50]
    all_ids = sum((manifest[k] for k in ("train", "val", "test", "id_test")), [])
    assert len(all_ids) == len(set(all_ids)) == 206
    assert set(manifest["train"]) | set(manifest["id_test"]) == set(source["train"])
    assert manifest["val"] == source["val"] and manifest["test"] == source["test"]
    assert manifest["holdout_checksum"] == holdout_checksum(manifest)


def main():
    ends = np.asarray(zarr.open(str(REPO / "data/pusht_cchi_v7_replay.zarr"), mode="r")["meta/episode_ends"][:])
    source = load_split_manifest(SOURCE, ends)
    shuffled = np.random.default_rng(42).permutation(source["train"])
    manifest = {k: source[k] for k in ("zarr_path", "n_episodes", "episode_ends_checksum")}
    manifest.update(generated_by="scripts/make_blq137_holdout.py",
        derivation=dict(source=str(SOURCE.relative_to(REPO)), source_checksum=source["checksum"],
            seed=42, method="Reserve first 50 entries of a seed-42 permutation of BLQ137 train as ID test; retain original OOD test and val."),
        train=sorted(map(int, shuffled[50:])), id_test=sorted(map(int, shuffled[:50])),
        val=source["val"], test=source["test"])
    lengths = np.diff(np.r_[0, ends])
    for key in ("train", "val", "test", "id_test"):
        manifest[key + "_frames"] = int(lengths[manifest[key]].sum())
    manifest["checksum"] = split_checksum(manifest["train"], manifest["val"], manifest["test"])
    manifest["holdout_checksum"] = holdout_checksum(manifest)
    validate_holdout(manifest, source)
    if DEST.exists() and json.loads(DEST.read_text()) != manifest:
        raise RuntimeError(f"Refusing to replace different split: {DEST}")
    DEST.write_text(json.dumps(manifest, indent=2) + "\n")
    load_split_manifest(DEST, ends, {"train": 87, "val": 19, "test": 50})
    print(f"Validated {DEST}: train=87, ID test=50, OOD test=50, val=19")


if __name__ == "__main__":
    main()
