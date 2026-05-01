"""WavLM-based model with a contrastive projection head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class WavLMSupConModel(nn.Module):
    """WavLM-base-plus encoder followed by a 2-layer projection head.

    The projection output is L2-normalised so that downstream contrastive
    loss reduces to a cosine-similarity formulation.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        projection_hidden: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        hidden_size = self.wavlm.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden, projection_dim),
        )

    def freeze_first_layers(self, n: int) -> None:
        """Freeze the first ``n`` transformer encoder layers."""
        for layer in self.wavlm.encoder.layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.wavlm(input_values=input_values)
        x = outputs.last_hidden_state.mean(dim=1)
        z = self.projection(x)
        z = F.normalize(z, dim=1)
        return z
