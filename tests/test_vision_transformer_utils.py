"""Tests for reusable Vision Transformer utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from build_a_text_to_image_generator_from_scratch_companion.utils.vision_transformer import (
    AttentionHead,
    Embeddings,
    PatchEmbeddings,
    VisionTransformerClassifier,
    ViTForClassfication,
)


@dataclass(frozen=True)
class SmallVisionConfig:
    """Minimal configuration for fast Vision Transformer tests."""

    image_size: int = 8
    patch_size: int = 4
    num_channels: int = 3
    hidden_size: int = 8
    num_attention_heads: int = 2
    intermediate_size: int = 16
    num_hidden_layers: int = 2
    num_classes: int = 5


def test_patch_embeddings_convert_spatial_grid_to_sequence() -> None:
    """An 8x8 image with 4x4 patches should produce four patch tokens."""
    module = PatchEmbeddings(SmallVisionConfig())
    images = torch.randn(2, 3, 8, 8)

    patches = module(images)

    assert patches.shape == (2, 4, 8)


def test_embeddings_prepend_one_class_token() -> None:
    """The class token should increase sequence length by one."""
    module = Embeddings(SmallVisionConfig())

    embeddings = module(torch.randn(2, 3, 8, 8))

    assert embeddings.shape == (2, 5, 8)


def test_attention_head_normalizes_each_query_distribution() -> None:
    """Every query should assign a probability distribution over all keys."""
    head = AttentionHead(hidden_size=8, attention_head_size=4)

    output, probabilities = head(torch.randn(2, 5, 8))

    assert output.shape == (2, 5, 4)
    assert probabilities.shape == (2, 5, 5)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((2, 5)))


def test_classifier_returns_logits_and_optional_attention_maps() -> None:
    """Classifier outputs should expose one attention map per encoder block."""
    config = SmallVisionConfig()
    model = VisionTransformerClassifier(config)
    images = torch.randn(2, 3, 8, 8)

    logits, attentions = model(images, output_attentions=True)

    assert logits.shape == (2, 5)
    assert attentions is not None
    assert len(attentions) == config.num_hidden_layers
    assert attentions[0].shape == (2, 2, 5, 5)


def test_upstream_misspelling_remains_a_compatibility_alias() -> None:
    """Existing book imports should continue to resolve."""
    assert ViTForClassfication is VisionTransformerClassifier
