"""Supervised Contrastive Loss (Khosla et al., 2020)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss as defined in Khosla et al., NeurIPS 2020.

    Operates on L2-normalised feature vectors. For every anchor in the batch
    the loss treats all samples sharing the anchor's label as positives, and
    all other samples as negatives.
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        self_mask = torch.ones_like(positive_mask) - torch.eye(batch_size, device=device)
        positive_mask = positive_mask * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_anchor = positive_mask.sum(dim=1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (
            positives_per_anchor + 1e-12
        )

        loss = -mean_log_prob_pos.mean()
        return loss
