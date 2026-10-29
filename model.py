"""BMSA-ViT model assembly and analytic complexity."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from bmsa_vit.layers import PatchEmbedding, PatchExtract, TransformerEncoder


def build_bmsa_vit(
    image_size: int = 48,
    patch_size: int = 4,
    embed_dim: int = 128,
    num_heads: int = 4,
    mlp_dim: int = 256,
    num_blocks: int = 4,
    num_classes: int = 2,
    attention_dropout: float = 0.1,
    projection_dropout: float = 0.1,
    mlp_dropout: float = 0.2,
    attention_type: str = "mha_self_attention",
    channels: int = 3,
) -> keras.Model:
    """Input [B, S, S, 3] in [0, 1] -> patches -> embedding -> encoders -> GAP -> softmax [B, K]."""
    num_patches = (image_size // patch_size) ** 2
    inputs = layers.Input(shape=(image_size, image_size, channels), name="input")
    x = PatchExtract(patch_size, name="patch_extract")(inputs)  # [B, N, P*P*C]
    x = PatchEmbedding(num_patches, embed_dim, name="patch_embedding")(x)  # [B, N, D]
    for i in range(num_blocks):
        x = TransformerEncoder(embed_dim, mlp_dim, num_heads, attention_dropout, projection_dropout,
                               mlp_dropout, attention_type, name=f"encoder_{i}")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)  # [B, D]  (Eq. 9)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="head")(x)
    return keras.Model(inputs, outputs, name="BMSA_ViT")


def count_params(image_size: int, patch_size: int, embed_dim: int = 128, mlp_dim: int = 256,
                 num_blocks: int = 4, num_classes: int = 2, channels: int = 3) -> int:
    """Closed-form trainable parameter count for attention_type='mha_self_attention'."""
    n = (image_size // patch_size) ** 2
    pdim = patch_size * patch_size * channels
    d = embed_dim
    embed = pdim * d + d + n * d
    block = 2 * (2 * d) + 4 * (d * d + d) + (d * mlp_dim + mlp_dim) + (mlp_dim * d + d)
    head = d * num_classes + num_classes
    return embed + num_blocks * block + head


def count_macs(image_size: int, patch_size: int, embed_dim: int = 128, mlp_dim: int = 256,
               num_blocks: int = 4, num_classes: int = 2, channels: int = 3) -> int:
    """Multiply-accumulates per image (matmuls only; softmax/LN/GELU/adds ignored).
    FLOPs ~= 2 x MACs under the usual convention."""
    n = (image_size // patch_size) ** 2
    d = embed_dim
    embed = n * patch_size * patch_size * channels * d
    proj = n * 4 * d * d               # Q, K, V, output projections
    attn = 2 * n * n * d               # Q K^T and A V, summed over heads
    mlp = n * 2 * d * mlp_dim
    return embed + num_blocks * (proj + attn + mlp) + d * num_classes
