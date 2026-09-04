"""Compact GPT-2 model blocks used by transformer-based image generation."""

__all__ = ["GPTConfig", "NewGELU", "CausalSelfAttention", "Block", "GPT"]

import math
from typing import Protocol

import torch
from torch import Tensor, nn


class GPTConfig(Protocol):
    """Configuration attributes required by the GPT modules."""

    n_embd: int
    attn_pdrop: float
    resid_pdrop: float
    block_size: int
    n_head: int
    vocab_size: int
    embd_pdrop: float
    n_layer: int


class NewGELU(nn.GELU):
    """Tanh-approximated GELU, as used by GPT-2."""

    def __init__(self) -> None:
        """Select PyTorch's tanh approximation; the shape of ``inputs`` is preserved."""
        super().__init__(approximate="tanh")


class CausalSelfAttention(nn.Module):
    """GPT-style masked multi-head self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize attention projections, dropout, and causal mask."""
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # Shape: (1, 1, block_size, block_size)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply causal self-attention to batch-first token states.

        Args:
            inputs: Token states shaped ``(batch_size, sequence_length, n_embd)``.

        Returns:
            Attended states shaped ``(batch_size, sequence_length, n_embd)``.
        """
        batch_size, sequence_length, embedding_dim = inputs.size()
        # (batch_size, sequence_length, n_embd) -> (batch_size, sequence_length, 3 * n_embd),
        # then split into three (batch_size, sequence_length, n_embd) tensors.
        query, key, value = self.c_attn(inputs).split(self.n_embd, dim=2)
        head_size: int = embedding_dim // self.n_head

        # (batch_size, sequence_length, embedding_dim) -> (batch_size, n_head, sequence_length, head_size)
        query = query.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, head_size).transpose(1, 2)

        # (batch_size, n_head, sequence_length, head_size) @ (batch_size, n_head, head_size, sequence_length)
        # -> (batch_size, n_head, sequence_length, sequence_length)
        scores: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(head_size)
        # Shape: (1, 1, sequence_length, sequence_length); broadcasts over batch and heads.
        causal_mask: Tensor = self.bias[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))
        probabilities: Tensor = self.attn_dropout(nn.functional.softmax(scores, dim=-1))
        # (batch_size, n_head, sequence_length, sequence_length) @ (batch_size, n_head, sequence_length, head_size)
        # -> (batch_size, n_head, sequence_length, head_size)
        attended: Tensor = probabilities @ value
        # (batch_size, n_head, sequence_length, head_size) -> (batch_size, sequence_length, n_head * head_size)
        # where n_head * head_size == embedding_dim.
        concatenated: Tensor = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        )
        return self.resid_dropout(self.c_proj(concatenated))


class Block(nn.Module):
    """Pre-normalized GPT attention and feed-forward residual block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize normalization, attention, and MLP layers."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            {
                "c_fc": nn.Linear(config.n_embd, 4 * config.n_embd),
                "c_proj": nn.Linear(4 * config.n_embd, config.n_embd),
                "act": NewGELU(),
                "dropout": nn.Dropout(config.resid_pdrop),
            }
        )

    def _feed_forward(self, inputs: Tensor) -> Tensor:
        """Apply the MLP while preserving upstream state-dictionary names.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, n_embd)``.

        Returns:
            States shaped ``(batch_size, sequence_length, n_embd)``.
        """
        # (batch_size, sequence_length, n_embd) -> (batch_size, sequence_length, 4 * n_embd)
        # -> (batch_size, sequence_length, n_embd)
        hidden: Tensor = self.mlp["c_fc"](inputs)
        hidden = self.mlp["act"](hidden)
        hidden = self.mlp["c_proj"](hidden)
        return self.mlp["dropout"](hidden)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply attention and MLP residual updates.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, n_embd)``.

        Returns:
            Updated states shaped ``(batch_size, sequence_length, n_embd)``.
        """
        hidden: Tensor = inputs + self.attn(self.ln_1(inputs))
        return hidden + self._feed_forward(self.ln_2(hidden))


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(self, config: GPTConfig) -> None:
        """Initialize token, position, transformer, and output layers."""
        super().__init__()
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.embd_pdrop),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * config.n_layer),
                )
        parameter_count: int = sum(parameter.numel() for parameter in self.transformer.parameters())
        print(f"number of parameters: {parameter_count / 1e6:.2f}M")

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Predict next-token logits and optionally cross-entropy loss.

        Args:
            idx: Token IDs shaped ``(batch_size, sequence_length)``.
            targets: Optional target IDs with the same shape. ``-1`` entries
                are ignored by the loss.

        Returns:
            Logits shaped ``(batch_size, sequence_length, vocab_size)`` and
            optional scalar loss.

        Raises:
            ValueError: If the input sequence exceeds ``block_size``.
        """
        _, sequence_length = idx.size()
        if sequence_length > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {sequence_length}; "
                f"block size is {self.block_size}"
            )
        # Shape: (1, sequence_length)
        positions: Tensor = torch.arange(
            sequence_length, dtype=torch.long, device=idx.device
        ).unsqueeze(0)
        # Shape: (batch_size, sequence_length, n_embd)
        token_embeddings: Tensor = self.transformer["wte"](idx)
        # Shape: (1, sequence_length, n_embd); broadcasts over the batch when added.
        position_embeddings: Tensor = self.transformer["wpe"](positions)
        # Shape stays (batch_size, sequence_length, n_embd) through every block.
        hidden: Tensor = self.transformer["drop"](token_embeddings + position_embeddings)
        for block in self.transformer["h"]:
            hidden = block(hidden)
        hidden = self.transformer["ln_f"](hidden)
        # (batch_size, sequence_length, n_embd) -> (batch_size, sequence_length, vocab_size)
        logits: Tensor = self.lm_head(hidden)

        loss: Tensor | None = None
        if targets is not None:
            # Flatten to (batch_size * sequence_length, vocab_size) and (batch_size * sequence_length,).
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss
