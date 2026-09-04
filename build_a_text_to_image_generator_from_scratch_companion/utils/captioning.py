"""Image-caption dataset and encoder-decoder transformer utilities."""

__all__ = [
    "ImageTransform",
    "FlickrD",
    "center_crop",
    "FlickrCaptionDataset",
    "extract_patches",
    "SinusoidalPosEmb",
    "AttentionBlock",
    "TransformerBlock",
    "Decoder",
    "VisionEncoder",
    "VisionEncoderDecoder",
]

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import transforms

ImageTransform = Callable[[Image.Image], Tensor]


def center_crop(image: Image.Image) -> Image.Image:
    """Crop a PIL image to its largest centered square.

    Args:
        image: Source image.

    Returns:
        A centered square crop.
    """
    width, height = image.size
    side_length: int = min(width, height)
    left: float = (width - side_length) / 2
    top: float = (height - side_length) / 2
    right: float = (width + side_length) / 2
    bottom: float = (height + side_length) / 2
    return image.crop((left, top, right, bottom))


class FlickrCaptionDataset(Dataset):
    """Pair Flickr images with padded next-token caption targets.

    Args:
        images: Image paths, one per caption group.
        captions: Tokenized captions grouped by image.
        word2idx: Vocabulary mapping from tokens to integer IDs.
        max_length: Number of input and target positions returned per sample.
        image_size: Square image size after preprocessing.
    """

    def __init__(
        self,
        images: Sequence[str | Path],
        captions: Sequence[Sequence[Sequence[str]]],
        word2idx: Mapping[str, int],
        max_length: int = 50,
        image_size: int = 128,
    ) -> None:
        self.images = [str(image) for image in images]
        self.captions = captions
        self.word2idx = word2idx
        self._max_len = max_length
        self.image_size = image_size
        self._image_transform: ImageTransform = self._construct_image_transform()
        self._data: list[tuple[str, Sequence[str]]] = self._create_input_label_mappings()
        self._dataset_size = len(self._data)
        self._start_idx = 1
        self._end_idx = 2
        self._pad_idx = 0
        self._UNK_idx = 3
        self._START_token = "<start>"
        self._END_token = "<end>"
        self._PAD_token = "<pad>"
        self._UNK_token = "<unk>"

    def _construct_image_transform(self) -> ImageTransform:
        """Create ImageNet-compatible tensor normalization."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return transforms.Compose([transforms.ToTensor(), normalize])

    def _group_captions(self) -> dict[str, Sequence[Sequence[str]]]:
        """Associate each image path with its tokenized captions."""
        return {self.images[index]: self.captions[index] for index in range(len(self.images))}

    def _create_input_label_mappings(self) -> list[tuple[str, Sequence[str]]]:
        """Flatten image-to-many-caption groups into individual examples."""
        return [
            (image_path, caption)
            for image_path, image_captions in self._group_captions().items()
            for caption in image_captions
        ]

    def _load_and_prepare_image(self, image_name: str) -> Tensor:
        """Load, center-crop, resize, and normalize one image.

        Args:
            image_name: Image path.

        Returns:
            Normalized image shaped ``(channels, image_size, image_size)``.
        """
        with Image.open(image_name) as source_image:
            resized = center_crop(source_image).resize((self.image_size, self.image_size))
            return self._image_transform(resized)

    def __len__(self) -> int:
        """Return the number of image-caption pairs."""
        return self._dataset_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return one image and its shifted, padded caption tensors.

        Args:
            index: Example index.

        Returns:
            Image shaped ``(channels, image_size, image_size)``, ``input_ids`` and
            ``target_ids`` shaped ``(max_length,)``, and a Boolean ``attention_mask``
            shaped ``(max_length,)`` that is ``True`` at real token positions.
        """
        image_path, raw_tokens = self._data[index]
        # Shape: (channels, image_size, image_size)
        image_tensor: Tensor = self._load_and_prepare_image(image_path)

        normalized_tokens: list[str] = [
            token.strip().lower() for token in raw_tokens[: self._max_len]
        ]
        bounded_tokens: list[str] = [
            self._START_token,
            *normalized_tokens,
            self._END_token,
        ]
        input_tokens: list[str] = bounded_tokens[:-1]
        target_tokens: list[str] = bounded_tokens[1:]
        sample_size: int = len(input_tokens)

        padding_size: int = self._max_len - sample_size
        if padding_size > 0:
            padding_tokens: list[str] = [self._PAD_token] * padding_size
            input_tokens.extend(padding_tokens)
            target_tokens.extend(padding_tokens)

        # input_ids and target_ids: each (max_length,) after padding.
        input_ids = torch.tensor(
            [self.word2idx.get(token, self._UNK_idx) for token in input_tokens],
            dtype=torch.long,
        )
        target_ids = torch.tensor(
            [self.word2idx.get(token, self._UNK_idx) for token in target_tokens],
            dtype=torch.long,
        )
        # Shape: (max_length,); True marks real tokens, False marks padding.
        attention_mask: Tensor = torch.zeros(self._max_len, dtype=torch.bool)
        attention_mask[:sample_size] = True
        return image_tensor, input_ids, target_ids, attention_mask


# Backward-compatible name from the upstream notebooks.
FlickrD = FlickrCaptionDataset


def extract_patches(image_tensor: Tensor, patch_size: int = 16) -> Tensor:
    """Convert images into a sequence of flattened non-overlapping patches.

    Args:
        image_tensor: Images shaped ``(batch_size, channels, height, width)``.
        patch_size: Height and width of each square patch.

    Returns:
        Patches shaped ``(batch_size, num_patches, patch_vector_size)``.
    """
    batch_size, channels, _, _ = image_tensor.size()
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # (batch_size, channels, height, width) -> (batch_size, channels * patch_size**2, num_patches)
    unfolded: Tensor = unfold(image_tensor)
    # transpose(1, 2) -> (batch_size, num_patches, channels * patch_size**2)
    return unfolded.transpose(1, 2).reshape(batch_size, -1, channels * patch_size * patch_size)


class SinusoidalPosEmb(nn.Module):
    """Generate sinusoidal embeddings for scalar sequence positions."""

    def __init__(self, dim: int) -> None:
        """Store the requested embedding dimension."""
        super().__init__()
        self.dim = dim

    def forward(self, positions: Tensor) -> Tensor:
        """Embed scalar positions into sinusoidal vectors.

        Args:
            positions: Positions shaped ``(sequence_length,)``.

        Returns:
            Embeddings shaped ``(sequence_length, dim)``.
        """
        half_dim: int = self.dim // 2
        scale: float = math.log(10_000) / (half_dim - 1)
        # Shape: (half_dim,)
        frequencies: Tensor = torch.exp(torch.arange(half_dim, device=positions.device) * -scale)
        # (sequence_length, 1) * (1, half_dim) -> (sequence_length, half_dim)
        phases: Tensor = positions[:, None] * frequencies[None, :]
        # Concatenate sin and cos halves: (sequence_length, dim).
        return torch.cat((phases.sin(), phases.cos()), dim=-1)


class AttentionBlock(nn.Module):
    """Batch-first multi-head attention with optional causal masking."""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4, masking: bool = True) -> None:
        """Initialize the attention layer."""
        super().__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )

    def forward(
        self,
        queries: Tensor,
        keys_and_values: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend from queries to a key/value sequence.

        Args:
            queries: Queries shaped ``(batch_size, query_length, hidden_size)``.
            keys_and_values: Keys and values shaped ``(batch_size, key_length, hidden_size)``.
            key_mask: Optional Boolean mask shaped ``(batch_size, key_length)``;
                ``True`` marks padding to ignore.

        Returns:
            Attended states shaped ``(batch_size, query_length, hidden_size)``.
        """
        sequence_length: int = queries.size(1)
        causal_mask: Tensor | None = None
        if self.masking:
            # Shape: (query_length, query_length); True above the diagonal blocks future positions.
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=queries.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        output, _ = self.multihead_attn(
            queries,
            keys_and_values,
            keys_and_values,
            attn_mask=causal_mask,
            key_padding_mask=key_mask,
        )
        # Shape: (batch_size, query_length, hidden_size)
        return output


class TransformerBlock(nn.Module):
    """Self-attention block with optional decoder cross-attention."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        decoder: bool = False,
        masking: bool = True,
    ) -> None:
        """Initialize attention, normalization, and MLP layers."""
        super().__init__()
        self.decoder = decoder
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size, num_heads, masking)
        if decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size, num_heads, masking=False)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        inputs: Tensor,
        input_key_mask: Tensor | None = None,
        cross_key_mask: Tensor | None = None,
        kv_cross: Tensor | None = None,
    ) -> Tensor:
        """Apply self-attention, optional cross-attention, and an MLP.

        Args:
            inputs: States shaped ``(batch_size, sequence_length, hidden_size)``.
            input_key_mask: Optional padding mask shaped ``(batch_size, sequence_length)``.
            cross_key_mask: Optional padding mask shaped ``(batch_size, cross_length)``.
            kv_cross: Cross-attention keys/values shaped
                ``(batch_size, cross_length, hidden_size)``; required for decoder blocks.

        Returns:
            Updated states shaped ``(batch_size, sequence_length, hidden_size)``.

        Raises:
            ValueError: If a decoder block is called without ``kv_cross``.
        """
        # Every sublayer preserves (batch_size, sequence_length, hidden_size).
        hidden: Tensor = self.norm1(self.attn1(inputs, inputs, key_mask=input_key_mask) + inputs)
        if self.decoder:
            if kv_cross is None:
                raise ValueError("decoder blocks require kv_cross")
            hidden = self.norm2(self.attn2(hidden, kv_cross, key_mask=cross_key_mask) + hidden)
        return self.norm_mlp(self.mlp(hidden) + hidden)


class Decoder(nn.Module):
    """Autoregressive caption decoder with image cross-attention."""

    def __init__(
        self,
        num_emb: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        """Initialize token embeddings, blocks, and output projection."""
        super().__init__()
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.embedding.weight.data *= 0.001
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, decoder=True) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(hidden_size, num_emb)

    def forward(
        self,
        input_seq: Tensor,
        encoder_output: Tensor,
        input_padding_mask: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict vocabulary logits for every caption position.

        Args:
            input_seq: Token IDs shaped ``(batch_size, sequence_length)``.
            encoder_output: Visual tokens shaped ``(batch_size, num_patches, hidden_size)``.
            input_padding_mask: Optional mask shaped ``(batch_size, sequence_length)``.
            encoder_padding_mask: Optional mask shaped ``(batch_size, num_patches)``.

        Returns:
            Logits shaped ``(batch_size, sequence_length, num_emb)``.
        """
        # (batch_size, sequence_length) -> (batch_size, sequence_length, hidden_size)
        token_embeddings: Tensor = self.embedding(input_seq)
        # Shape: (sequence_length,)
        positions: Tensor = torch.arange(input_seq.size(1), device=input_seq.device)
        # Position embeddings (sequence_length, hidden_size) broadcast over the batch dimension.
        hidden: Tensor = token_embeddings + self.pos_emb(positions)
        for block in self.blocks:
            hidden = block(
                hidden,
                input_key_mask=input_padding_mask,
                cross_key_mask=encoder_padding_mask,
                kv_cross=encoder_output,
            )
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, num_emb)
        return self.fc_out(hidden)


class VisionEncoder(nn.Module):
    """Encode image patches into contextual visual tokens."""

    def __init__(
        self,
        image_size: int,
        channels_in: int,
        patch_size: int = 16,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        """Initialize patch projection, positions, and encoder blocks."""
        super().__init__()
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        sequence_length: int = (image_size // patch_size) ** 2
        # Shape: (1, num_patches, hidden_size); broadcasts over the batch in ``forward``.
        self.pos_embedding = nn.Parameter(
            torch.empty(1, sequence_length, hidden_size).normal_(std=0.02)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, decoder=False, masking=False)
                for _ in range(num_layers)
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        """Return contextual patch embeddings for an image batch.

        Args:
            image: Images shaped ``(batch_size, channels_in, image_size, image_size)``.

        Returns:
            Visual tokens shaped ``(batch_size, num_patches, hidden_size)`` with
            ``num_patches = (image_size / patch_size) ** 2``.
        """
        # (batch_size, channels_in, image_size, image_size) -> (batch_size, num_patches, channels_in * patch_size**2)
        patches: Tensor = extract_patches(image, patch_size=self.patch_size)
        # (batch_size, num_patches, channels_in * patch_size**2) -> (batch_size, num_patches, hidden_size)
        hidden: Tensor = self.fc_in(patches) + self.pos_embedding
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class VisionEncoderDecoder(nn.Module):
    """Image encoder and autoregressive caption decoder."""

    def __init__(
        self,
        image_size: int,
        channels_in: int,
        num_emb: int,
        patch_size: int = 16,
        hidden_size: int = 128,
        num_layers: tuple[int, int] = (3, 3),
        num_heads: int = 4,
    ) -> None:
        """Initialize the visual encoder and caption decoder."""
        super().__init__()
        encoder_layers, decoder_layers = num_layers
        self.encoder = VisionEncoder(
            image_size=image_size,
            channels_in=channels_in,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            num_heads=num_heads,
        )
        self.decoder = Decoder(
            num_emb=num_emb,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            num_heads=num_heads,
        )

    def forward(self, input_image: Tensor, target_seq: Tensor, padding_mask: Tensor) -> Tensor:
        """Predict caption logits conditioned on image patches.

        Args:
            input_image: Images shaped ``(batch_size, channels_in, image_size, image_size)``.
            target_seq: Caption token IDs shaped ``(batch_size, sequence_length)``.
            padding_mask: Mask shaped ``(batch_size, sequence_length)``; non-zero marks
                real tokens.

        Returns:
            Logits shaped ``(batch_size, sequence_length, num_emb)``.
        """
        # (batch_size, sequence_length): True marks padding positions to ignore.
        bool_padding_mask: Tensor = padding_mask == 0
        # Shape: (batch_size, num_patches, hidden_size)
        encoded_sequence: Tensor = self.encoder(input_image)
        return self.decoder(
            input_seq=target_seq,
            encoder_output=encoded_sequence,
            input_padding_mask=bool_padding_mask,
        )
