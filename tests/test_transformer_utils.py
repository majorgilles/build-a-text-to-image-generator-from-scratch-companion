"""Tests for reusable encoder-decoder transformer utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from build_a_text_to_image_generator_from_scratch_companion.utils.transformer import (
    Batch,
    LabelSmoothing,
    MultiHeadedAttention,
    NoamOpt,
    attention,
    create_model,
    make_std_mask,
    subsequent_mask,
)


def test_subsequent_mask_hides_future_positions() -> None:
    """Only the current and earlier positions should be visible."""
    expected = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )

    assert torch.equal(subsequent_mask(3), expected)


def test_make_std_mask_combines_padding_and_causal_masks() -> None:
    """Padding tokens and future positions should both be hidden."""
    target = torch.tensor([[4, 5, 0]])

    mask = make_std_mask(target, padding_id=0)

    expected = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, False]]]
    )
    assert torch.equal(mask, expected)


def test_batch_shifts_targets_for_teacher_forcing() -> None:
    """Batch should split target sequences into input and next-token labels."""
    source = np.array([[2, 3, 0]], dtype=np.int64)
    target = np.array([[1, 4, 5, 2]], dtype=np.int64)

    batch = Batch(source, target, device="cpu")

    assert torch.equal(batch.src, torch.tensor([[2, 3, 0]]))
    assert torch.equal(batch.trg, torch.tensor([[1, 4, 5]]))
    assert torch.equal(batch.trg_y, torch.tensor([[4, 5, 2]]))
    assert batch.ntokens is not None
    assert batch.ntokens.item() == 3


def test_scaled_dot_product_attention_has_expected_shape_and_probabilities() -> None:
    """Attention probabilities should normalize over key positions."""
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    values, probabilities = attention(query, query, query)

    assert values.shape == (1, 2, 2)
    assert probabilities.shape == (1, 2, 2)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones((1, 2))
    )


def test_multihead_attention_rejects_incompatible_dimensions() -> None:
    """Each attention head must receive the same integer-sized projection."""
    with pytest.raises(ValueError, match="d_model must be divisible"):
        MultiHeadedAttention(h=3, d_model=8)


def test_create_model_runs_an_encoder_decoder_forward_pass() -> None:
    """The assembled transformer should preserve batch and sequence dimensions."""
    torch.manual_seed(7)
    model = create_model(
        src_vocab=11,
        tgt_vocab=13,
        num_layers=2,
        d_model=8,
        d_ff=16,
        num_heads=2,
        dropout=0.0,
        device="cpu",
    )
    source = torch.tensor([[2, 3, 4]])
    target = torch.tensor([[1, 5]])

    output = model(
        source,
        target,
        torch.ones((1, 1, 3), dtype=torch.bool),
        subsequent_mask(2),
    )

    assert output.shape == (1, 2, 8)
    assert model.generator(output).shape == (1, 2, 13)


def test_label_smoothing_zeros_padding_rows() -> None:
    """Padding targets should contribute an all-zero target distribution."""
    criterion = LabelSmoothing(size=5, padding_idx=0, smoothing=0.1)
    predictions = torch.log_softmax(torch.randn(2, 5), dim=-1)

    loss = criterion(predictions, torch.tensor([3, 0]))

    assert loss.ndim == 0
    assert criterion.true_dist is not None
    assert torch.equal(criterion.true_dist[1], torch.zeros(5))


def test_noam_optimizer_rate_matches_transformer_schedule() -> None:
    """The warmup schedule should match its closed-form equation."""
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.0)
    schedule = NoamOpt(model_size=512, factor=1.0, warmup=4_000, optimizer=optimizer)

    assert schedule.rate(1) == pytest.approx(512**-0.5 * 4_000**-1.5)
