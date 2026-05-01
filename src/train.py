"""Training loop for WavLM Supervised Contrastive fine-tuning on AbjadKids."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import (
    AbjadAudioDataset,
    BalancedBatchSampler,
    collate_fn,
)
from src.loss import SupConLoss
from src.model import WavLMSupConModel


@dataclass
class TrainConfig:
    backbone: str = "microsoft/wavlm-base-plus"
    projection_hidden: int = 256
    projection_dim: int = 128
    dropout: float = 0.1
    freeze_first_n_layers: int = 4

    sample_rate: int = 16000
    max_seconds: float = 2.5

    n_classes_per_batch: int = 8
    n_samples_per_class: int = 4

    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.98)
    grad_clip_norm: float = 1.0

    total_steps: int = 2000
    eta_min: float = 1e-6
    temperature: float = 0.05

    print_every: int = 10
    save_every: int = 100

    checkpoint_dir: str = "checkpoints"


def save_checkpoint(model, optimizer, scheduler, step, label2id, id2label, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "label2id": label2id,
            "id2label": id2label,
        },
        path,
    )


def train(train_df, label2id, id2label, cfg: TrainConfig, device: str = "cuda"):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    max_len = int(cfg.sample_rate * cfg.max_seconds)
    train_dataset = AbjadAudioDataset(train_df, sr=cfg.sample_rate, max_len=max_len)
    sampler = BalancedBatchSampler(
        train_df,
        n_classes=cfg.n_classes_per_batch,
        n_samples=cfg.n_samples_per_class,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    model = WavLMSupConModel(
        model_name=cfg.backbone,
        projection_hidden=cfg.projection_hidden,
        projection_dim=cfg.projection_dim,
        dropout=cfg.dropout,
    ).to(device)
    model.freeze_first_layers(cfg.freeze_first_n_layers)

    criterion = SupConLoss(temperature=cfg.temperature)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.total_steps, eta_min=cfg.eta_min
    )

    model.train()
    loader_iter = iter(train_loader)
    pbar = tqdm(range(cfg.total_steps), total=cfg.total_steps)

    for step in pbar:
        batch = next(loader_iter)
        input_values = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(input_values)
        loss = criterion(embeddings, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % cfg.print_every == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_description(f"Step {step} | Loss {loss.item():.4f} | LR {lr:.2e}")

        if step > 0 and step % cfg.save_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"wavlm_supcon_step_{step}.pt")
            save_checkpoint(model, optimizer, scheduler, step, label2id, id2label, path)

    final_path = os.path.join(cfg.checkpoint_dir, f"wavlm_supcon_step_{cfg.total_steps}.pt")
    save_checkpoint(
        model, optimizer, scheduler, cfg.total_steps, label2id, id2label, final_path
    )
    return model
