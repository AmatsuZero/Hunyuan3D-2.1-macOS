"""Torch-free DINO token projection for the outer paint UNet."""

import mlx.nn as nn


class MLXDinoProjector(nn.Module):
    def __init__(self, input_dim=1536, output_dim=1024, extra_tokens=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_tokens = extra_tokens
        self.proj = nn.Linear(input_dim, extra_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def __call__(self, image_embeds):
        if image_embeds.ndim not in (2, 3) or image_embeds.shape[-1] != self.input_dim:
            raise ValueError("expected DINO embeddings with shape [B, C] or [B, N, C]")
        batch = image_embeds.shape[0]
        projected = self.proj(image_embeds.reshape(-1, self.input_dim))
        projected = self.norm(projected.reshape(-1, self.extra_tokens, self.output_dim))
        return projected.reshape(batch, -1, self.output_dim)
