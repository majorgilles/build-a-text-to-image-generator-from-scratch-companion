"""Typed patch embedding, attention, encoder, and classification modules used by Chapter 3."""

__all__ = [
    "ViTForClassfication",
    "VisionTransformerConfig",
    "GELU",
    "PatchEmbeddings",
    "Embeddings",
    "AttentionHead",
    "MultiHeadAttention",
    "MLP",
    "Block",
    "Encoder",
    "VisionTransformerClassifier",
]

import math
from typing import Protocol

import torch
from torch import Tensor, nn


class VisionTransformerConfig(Protocol):
    """Configuration attributes required by the Vision Transformer modules."""

    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    num_hidden_layers: int
    num_classes: int


class GELU(nn.GELU):
    """Tanh-approximated GELU, as used by the original ViT implementation."""

    def __init__(self) -> None:
        """Select PyTorch's tanh approximation; the shape of ``inputs`` is preserved."""
        super().__init__(approximate="tanh")


class PatchEmbeddings(nn.Module):
    """Split images into non-overlapping patches and project each patch."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize the strided patch projection."""
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Convert images to a sequence of patch embeddings.

        Args:
            images: Images shaped ``(batch_size, num_channels, image_size, image_size)``.

        Returns:
            Patch embeddings shaped ``(batch_size, num_patches, hidden_size)`` with
            ``num_patches = (image_size / patch_size) ** 2``.
        """
        # (batch_size, channels, height, width) -> (batch_size, hidden_size, grid_height, grid_width)
        # where grid_height = height / patch_size and grid_width = width / patch_size.
        projected: Tensor = self.projection(images)
        # flatten(2): (batch_size, hidden_size, num_patches) with num_patches = grid_height * grid_width;
        # transpose(1, 2): (batch_size, num_patches, hidden_size).
        return projected.flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    """Combine patch embeddings, a class token, and learned positions."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize patch, class-token, and positional embeddings."""
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Shape: (1, 1, hidden_size); expanded over the batch in ``forward``.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        num_patches: int = (config.image_size // config.patch_size) ** 2
        # Shape: (1, num_patches + 1, hidden_size); the +1 is the class-token position.
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))

    def forward(self, images: Tensor) -> Tensor:
        """Return patch sequence embeddings with a prepended class token.

        Args:
            images: Images shaped ``(batch_size, num_channels, image_size, image_size)``.

        Returns:
            Sequence embeddings shaped ``(batch_size, num_patches + 1, hidden_size)``.
        """
        # Shape: (batch_size, num_patches, hidden_size)
        patch_embeddings: Tensor = self.patch_embeddings(images)
        batch_size: int = patch_embeddings.size(0)
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        class_tokens: Tensor = self.cls_token.expand(batch_size, -1, -1)
        # Prepend along the sequence axis: (batch_size, num_patches + 1, hidden_size).
        sequence: Tensor = torch.cat((class_tokens, patch_embeddings), dim=1)
        # Position embeddings (1, num_patches + 1, hidden_size) broadcast over the batch.
        return sequence + self.position_embeddings


class AttentionHead(nn.Module):
    """One scaled dot-product self-attention head."""

    def __init__(self, hidden_size: int, attention_head_size: int, bias: bool = True) -> None:
        """Initialize query, key, and value projections."""
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return attended values and attention probabilities.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, hidden_size)``.

        Returns:
            Attended values shaped ``(batch_size, sequence_length, attention_head_size)``
            and probabilities shaped ``(batch_size, sequence_length, sequence_length)``.
        """
        # Each: (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query: Tensor = self.query(inputs)
        key: Tensor = self.key(inputs)
        value: Tensor = self.value(inputs)
        # (batch_size, sequence_length, head) @ (batch_size, head, sequence_length)
        # -> (batch_size, sequence_length, sequence_length)
        scores: Tensor = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.attention_head_size)
        probabilities: Tensor = nn.functional.softmax(scores, dim=-1)
        # (batch_size, sequence_length, sequence_length) @ (batch_size, sequence_length, head)
        # -> (batch_size, sequence_length, attention_head_size)
        return probabilities @ value, probabilities


class MultiHeadAttention(nn.Module):
    """Independent attention heads followed by an output projection."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize all attention heads."""
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = nn.ModuleList(
            [
                AttentionHead(self.hidden_size, self.attention_head_size)
                for _ in range(self.num_attention_heads)
            ]
        )
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(
        self, inputs: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Run all heads and optionally return their attention maps.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, hidden_size)``.
            output_attentions: Whether to also return the stacked attention maps.

        Returns:
            Output shaped ``(batch_size, sequence_length, hidden_size)`` and, when
            requested, probabilities shaped
            ``(batch_size, num_attention_heads, sequence_length, sequence_length)``.
        """
        head_outputs: list[tuple[Tensor, Tensor]] = [head(inputs) for head in self.heads]
        # Concatenate heads along the feature axis: (batch_size, sequence_length, all_head_size).
        concatenated: Tensor = torch.cat(
            [attention_output for attention_output, _ in head_outputs], dim=-1
        )
        # (batch_size, sequence_length, all_head_size) -> (batch_size, sequence_length, hidden_size)
        output: Tensor = self.output_projection(concatenated)
        if not output_attentions:
            return output, None
        # Stack per-head maps: (batch_size, num_attention_heads, sequence_length, sequence_length).
        probabilities: Tensor = torch.stack(
            [attention_probs for _, attention_probs in head_outputs], dim=1
        )
        return output, probabilities


class MLP(nn.Module):
    """Two-layer feed-forward network used inside a ViT block."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize feed-forward projections and activation."""
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project to the intermediate size and back to hidden size.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, hidden_size)``.

        Returns:
            States shaped ``(batch_size, sequence_length, hidden_size)``.
        """
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, intermediate_size)
        # -> (batch_size, sequence_length, hidden_size)
        hidden: Tensor = self.activation(self.dense_1(inputs))
        return self.dense_2(hidden)


class Block(nn.Module):
    """Pre-normalized self-attention and MLP residual block."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize attention, normalization, and MLP modules."""
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config.hidden_size)

    def forward(
        self, inputs: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Apply attention and feed-forward residual updates.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, hidden_size)``.
            output_attentions: Whether to also return this block's attention maps.

        Returns:
            Updated states shaped ``(batch_size, sequence_length, hidden_size)`` and
            optional probabilities shaped
            ``(batch_size, num_attention_heads, sequence_length, sequence_length)``.
        """
        # Both residual branches preserve (batch_size, sequence_length, hidden_size).
        attention_output, attention_probs = self.attention(
            self.layernorm_1(inputs), output_attentions=output_attentions
        )
        hidden: Tensor = inputs + attention_output
        hidden = hidden + self.mlp(self.layernorm_2(hidden))
        return hidden, attention_probs


class Encoder(nn.Module):
    """Stack of Vision Transformer blocks."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize the configured number of blocks."""
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self, inputs: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Encode patch embeddings and optionally collect attention maps.

        Args:
            inputs: Embeddings shaped ``(batch_size, num_patches + 1, hidden_size)``.
            output_attentions: Whether to collect every block's attention maps.

        Returns:
            Encoded states shaped ``(batch_size, num_patches + 1, hidden_size)`` and an
            optional list of ``num_hidden_layers`` maps, each shaped
            ``(batch_size, num_attention_heads, num_patches + 1, num_patches + 1)``.
        """
        # Shape stays (batch_size, num_patches + 1, hidden_size) through every block.
        hidden: Tensor = inputs
        all_attentions: list[Tensor] = []
        for block in self.blocks:
            hidden, attention_probs = block(hidden, output_attentions=output_attentions)
            if attention_probs is not None:
                all_attentions.append(attention_probs)
        return hidden, all_attentions if output_attentions else None


class VisionTransformerClassifier(nn.Module):
    """Vision Transformer encoder with a class-token classification head."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        """Initialize embeddings, encoder, and classifier."""
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.num_classes = config.num_classes
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(
        self, images: Tensor, output_attentions: bool = False
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Classify images and optionally return block attention maps.

        Args:
            images: Images shaped ``(batch_size, num_channels, image_size, image_size)``.
            output_attentions: Whether to return every block's attention maps.

        Returns:
            Logits shaped ``(batch_size, num_classes)`` and an optional list of
            ``num_hidden_layers`` maps, each shaped
            ``(batch_size, num_attention_heads, num_patches + 1, num_patches + 1)``.
        """
        # (batch_size, channels, image_size, image_size) -> (batch_size, num_patches + 1, hidden_size)
        embeddings: Tensor = self.embedding(images)
        # Shape: (batch_size, num_patches + 1, hidden_size)
        encoded, all_attentions = self.encoder(embeddings, output_attentions=output_attentions)
        # Class token at position 0: (batch_size, hidden_size) -> (batch_size, num_classes)
        logits: Tensor = self.classifier(encoded[:, 0, :])
        return logits, all_attentions

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize modules with the original ViT parameter scheme."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.position_embeddings.dtype)
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=0.02,
            ).to(module.cls_token.dtype)


# Compatibility with the misspelled class name in the upstream notebooks.
ViTForClassfication = VisionTransformerClassifier
