"""Download AbjadKids and build speaker-disjoint train / val / test splits."""

from __future__ import annotations

import argparse
import os

import yaml
from huggingface_hub import snapshot_download

from src.dataset import build_splits, index_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="data/")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.makedirs(args.output_dir, exist_ok=True)

    local_dir = cfg["data"]["local_dir"]
    print(f"Downloading {cfg['data']['dataset_id']} -> {local_dir}")
    snapshot_download(
        repo_id=cfg["data"]["dataset_id"],
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=16,
    )

    print("Indexing audio files...")
    df = index_dataset(
        data_dir=local_dir,
        categories=cfg["data"]["categories"],
        audio_extensions=cfg["data"]["audio_extensions"],
    )
    print(f"  Found {len(df)} files across {df['label_name'].nunique()} classes.")

    splits = build_splits(
        df,
        train_size=cfg["data"]["speaker_disjoint_split"]["train"],
        val_size=cfg["data"]["speaker_disjoint_split"]["val"],
        test_size=cfg["data"]["speaker_disjoint_split"]["test"],
        seed=cfg["data"]["speaker_disjoint_split"]["seed"],
        excluded_classes=cfg["data"].get("excluded_classes"),
    )

    for name in ("train", "val", "test"):
        out_path = os.path.join(args.output_dir, f"{name}.csv")
        splits[name].to_csv(out_path, index=False)
        print(
            f"  {name}: {len(splits[name])} files, "
            f"{splits[name]['speaker'].nunique()} speakers, "
            f"{splits[name]['label_name'].nunique()} labels -> {out_path}"
        )

    overlap_train_val = set(splits["train"]["speaker"]) & set(splits["val"]["speaker"])
    overlap_train_test = set(splits["train"]["speaker"]) & set(splits["test"]["speaker"])
    overlap_val_test = set(splits["val"]["speaker"]) & set(splits["test"]["speaker"])
    assert not (overlap_train_val or overlap_train_test or overlap_val_test), (
        "Speaker overlap detected between splits."
    )
    print("Speaker-disjoint splits verified.")


if __name__ == "__main__":
    main()
