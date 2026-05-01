"""End-to-end entry point for WavLM Supervised Contrastive fine-tuning."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import yaml

from src.train import TrainConfig, train


def cfg_from_yaml(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(
        backbone=raw["model"]["backbone"],
        projection_hidden=raw["model"]["projection_hidden"],
        projection_dim=raw["model"]["projection_dim"],
        dropout=raw["model"]["dropout"],
        freeze_first_n_layers=raw["model"]["freeze_first_n_layers"],
        sample_rate=raw["audio"]["sample_rate"],
        max_seconds=raw["audio"]["max_seconds"],
        n_classes_per_batch=raw["sampler"]["n_classes_per_batch"],
        n_samples_per_class=raw["sampler"]["n_samples_per_class"],
        learning_rate=raw["optim"]["learning_rate"],
        weight_decay=raw["optim"]["weight_decay"],
        betas=tuple(raw["optim"]["betas"]),
        grad_clip_norm=raw["optim"]["grad_clip_norm"],
        total_steps=raw["schedule"]["total_steps"],
        eta_min=raw["schedule"]["eta_min"],
        temperature=raw["loss"]["temperature"],
        print_every=raw["logging"]["print_every"],
        save_every=raw["logging"]["save_every"],
        checkpoint_dir=raw["paths"]["checkpoint_dir"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_csv", default="data/train.csv")
    args = parser.parse_args()

    cfg = cfg_from_yaml(args.config)
    train_df = pd.read_csv(args.train_csv)

    label_names = sorted(train_df["label_name"].unique())
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(train_df, label2id, id2label, cfg, device=device)


if __name__ == "__main__":
    main()
