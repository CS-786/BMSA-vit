"""BMSA-ViT layers.

Shape notation: B = batch, H = W = image size, P = patch size, C = 3,
N = (H/P)^2 tokens, D = embed_dim, h = num_heads.

The computation follows the reference notebook (01-idc_notebook.ipynb). The
notebook built attention from raw tf ops on symbolic tensors, which Keras 3
does not allow; here each block is a Layer subclass with the same weights
and the same arithmetic.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="bmsa_vit")
class PatchExtract(layers.Layer):
    """Non-overlapping P x P patches. [B, H, W, C] -> [B, N, P*P*C] (raster order)."""

    def __init__(self, patch_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)

    def build(self, input_shape) -> None:
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"Image size {h}x{w} is not divisible by patch size {self.patch_size}")
        self.num_patches = (h // self.patch_size) * (w // self.patch_size)
        self.patch_dim = self.patch_size * self.patch_size * c
        super().build(input_shape)

    def call(self, images: tf.Tensor) -> tf.Tensor:
        p = self.patch_size
        patches = tf.image.extract_patches(
            images=images, sizes=(1, p, p, 1), strides=(1, p, p, 1),
            rates=(1, 1, 1, 1), padding="VALID",
        )  # [B, H/P, W/P, P*P*C]
        return tf.reshape(patches, (-1, self.num_patches, self.patch_dim))  # [B, N, P*P*C]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_patches, self.patch_dim)

    def get_config(self) -> dict:
        return super().get_config() | {"patch_size": self.patch_size}


@keras.utils.register_keras_serializable(package="bmsa_vit")
class PatchEmbedding(layers.Layer):
    """Linear projection plus learned 1-D position embedding. [B, N, P*P*C] -> [B, N, D] (Eq. 4)."""

    def __init__(self, num_patches: int, embed_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.embed_dim = int(embed_dim)
        self.proj = layers.Dense(self.embed_dim, name="proj")
        self.pos_embed = layers.Embedding(input_dim=self.num_patches, output_dim=self.embed_dim, name="pos_embed")

    def call(self, patches: tf.Tensor) -> tf.Tensor:
        pos = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.proj(patches) + self.pos_embed(pos)  # [B, N, D] + [N, D]

    def get_config(self) -> dict:
        return super().get_config() | {"num_patches": self.num_patches, "embed_dim": self.embed_dim}


@keras.utils.register_keras_serializable(package="bmsa_vit")
class MultiHeadSelfAttention(layers.Layer):
    """Scaled dot-product multi-head self-attention. [B, N, D] -> [B, N, D].

        Q, K, V = x W_q, x W_k, x W_v           each [B, h, N, D/h]
        A = softmax(Q K^T / sqrt(D/h))          [B, h, N, N]
        y = Dropout_p( concat_h(Dropout_a(A) V) W_o )
    """

    def __init__(self, dim: int, num_heads: int, attention_dropout: float = 0.0,
                 projection_dropout: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.attention_dropout = float(attention_dropout)
        self.projection_dropout = float(projection_dropout)
        self.q = layers.Dense(self.dim, name="query")
        self.k = layers.Dense(self.dim, name="key")
        self.v = layers.Dense(self.dim, name="value")
        self.out = layers.Dense(self.dim, name="out")
        self.attn_drop = layers.Dropout(self.attention_dropout)
        self.proj_drop = layers.Dropout(self.projection_dropout)

    def _split(self, t: tf.Tensor, b: tf.Tensor, n: tf.Tensor) -> tf.Tensor:
        t = tf.reshape(t, (b, n, self.num_heads, self.head_dim))
        return tf.transpose(t, perm=[0, 2, 1, 3])  # [B, h, N, D/h]

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        b, n = tf.shape(x)[0], tf.shape(x)[1]
        q = self._split(self.q(x), b, n)
        k = self._split(self.k(x), b, n)
        v = self._split(self.v(x), b, n)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, q.dtype))  # [B, h, N, N]
        attn = self.attn_drop(tf.nn.softmax(scores, axis=-1), training=training)
        y = tf.matmul(attn, v)  # [B, h, N, D/h]
        y = tf.reshape(tf.transpose(y, perm=[0, 2, 1, 3]), (b, n, self.dim))  # [B, N, D]
        return self.proj_drop(self.out(y), training=training)

    def get_config(self) -> dict:
        return super().get_config() | {
            "dim": self.dim, "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout, "projection_dropout": self.projection_dropout,
        }


@keras.utils.register_keras_serializable(package="bmsa_vit")
class TransformerEncoder(layers.Layer):
    """Pre-norm encoder block. [B, N, D] -> [B, N, D].

        x = x + MHSA(LN(x));  x = x + MLP(LN(x)),  MLP = Dense(D) . Drop . GELU-Dense(mlp_dim) . Drop  (Eq. 8)

    attention_type="mha_layer" uses keras.layers.MultiHeadAttention with
    key_dim = embed_dim per head, exactly as the notebook's alternative branch
    (this gives 4x the attention parameters of "mha_self_attention" for 4 heads).
    """

    def __init__(self, embed_dim: int, mlp_dim: int, num_heads: int, attention_dropout: float = 0.1,
                 projection_dropout: float = 0.1, mlp_dropout: float = 0.2,
                 attention_type: str = "mha_self_attention", **kwargs) -> None:
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.mlp_dim = int(mlp_dim)
        self.num_heads = int(num_heads)
        self.attention_dropout = float(attention_dropout)
        self.projection_dropout = float(projection_dropout)
        self.mlp_dropout = float(mlp_dropout)
        self.attention_type = attention_type
        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name="norm2")
        if attention_type == "mha_self_attention":
            self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout, projection_dropout, name="mhsa")
        elif attention_type == "mha_layer":
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                                  dropout=attention_dropout, name="mha")
        else:
            raise ValueError(f"Unknown attention_type '{attention_type}'")
        self.fc1 = layers.Dense(mlp_dim, activation=tf.nn.gelu, name="mlp_fc1")
        self.fc2 = layers.Dense(embed_dim, name="mlp_fc2")
        self.drop1 = layers.Dropout(mlp_dropout)
        self.drop2 = layers.Dropout(mlp_dropout)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        h = self.norm1(x)
        if self.attention_type == "mha_layer":
            h = self.attn(h, h, training=training)
        else:
            h = self.attn(h, training=training)
        x = x + h
        h = self.norm2(x)
        h = self.drop1(self.fc1(h), training=training)
        h = self.drop2(self.fc2(h), training=training)
        return x + h

    def get_config(self) -> dict:
        return super().get_config() | {
            "embed_dim": self.embed_dim, "mlp_dim": self.mlp_dim, "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout, "projection_dropout": self.projection_dropout,
            "mlp_dropout": self.mlp_dropout, "attention_type": self.attention_type,
        }
