"""Contrastive Language-Image Pretraining data and model utilities."""

__all__ = [
    "TokenizerLike",
    "ImageTransformLike",
    "CFG",
    "AvgMeter",
    "get_lr",
    "CLIPDataset",
    "get_transforms",
    "ImageEncoder",
    "TextEncoder",
    "ProjectionHead",
    "CLIPModel",
    "cross_entropy",
]

from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal, Protocol

import albumentations as A
import cv2
import numpy as np
import timm
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from transformers import DistilBertConfig, DistilBertModel


class TokenizerLike(Protocol):
    """Tokenizer interface required by :class:`CLIPDataset`."""

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
    ) -> Mapping[str, Sequence[Sequence[int]]]:
        """Tokenize a text batch."""


class ImageTransformLike(Protocol):
    """Albumentations-style image transform interface."""

    def __call__(self, *, image: NDArray[np.uint8]) -> Mapping[str, NDArray]:
        """Transform one ``(height, width, channels)`` image supplied by keyword.

        Returns:
            Mapping whose ``image`` entry is shaped ``(size, size, channels)``.
        """


class CFG:
    """Default hyperparameters retained for upstream notebook compatibility."""

    image_path: ClassVar[str] = "files/Images"
    captions_path: ClassVar[str] = "files"
    batch_size: ClassVar[int] = 32
    head_lr: ClassVar[float] = 1e-3
    weight_decay: ClassVar[float] = 1e-3
    patience: ClassVar[int] = 1
    factor: ClassVar[float] = 0.8
    epochs: ClassVar[int] = 4
    device: ClassVar[str] = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: ClassVar[str] = "resnet50"
    image_embedding: ClassVar[int] = 2_048
    text_encoder_model: ClassVar[str] = "distilbert-base-uncased"
    text_embedding: ClassVar[int] = 768
    text_tokenizer: ClassVar[str] = "distilbert-base-uncased"
    max_length: ClassVar[int] = 200
    pretrained: ClassVar[bool] = True
    trainable: ClassVar[bool] = False
    temperature: ClassVar[float] = 1.0
    size: ClassVar[int] = 224
    num_projection_layers: ClassVar[int] = 1
    projection_dim: ClassVar[int] = 256
    dropout: ClassVar[float] = 0.1


class AvgMeter:
    """Track the weighted running average of a scalar metric."""

    def __init__(self, name: str = "Metric") -> None:
        """Initialize and reset the meter."""
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset the average, sum, and observation count."""
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, count: int = 1) -> None:
        """Add a value repeated ``count`` times to the running average."""
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        """Return a compact metric display string."""
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate of the optimizer's first parameter group."""
    return float(optimizer.param_groups[0]["lr"])


class CLIPDataset(torch.utils.data.Dataset):
    """Load paired image-caption examples for CLIP training.

    Args:
        image_filenames: Image filenames relative to ``CFG.image_path``.
        captions: Captions aligned with the image filenames.
        tokenizer: Batch tokenizer returning input IDs and attention masks.
        transforms: Albumentations-style image transform.
    """

    def __init__(
        self,
        image_filenames: Sequence[str],
        captions: Sequence[str],
        tokenizer: TokenizerLike,
        transforms: ImageTransformLike,
    ) -> None:
        self.image_filenames = list(image_filenames)
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            self.captions,
            padding=True,
            truncation=True,
            max_length=CFG.max_length,
        )
        self.transforms = transforms

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        """Return encoded text, a normalized image, and the raw caption.

        Args:
            index: Example index.

        Returns:
            Mapping with ``input_ids`` and ``attention_mask`` shaped ``(padded_length,)``,
            ``image`` shaped ``(channels, size, size)``, and the ``caption`` string.
        """
        # input_ids and attention_mask: each (padded_length,) for this caption.
        item: dict[str, Tensor | str] = {
            key: torch.tensor(values[index]) for key, values in self.encoded_captions.items()
        }
        image_path = f"{CFG.image_path}/{self.image_filenames[index]}"
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        # Shape: (height, width, channels) uint8
        rgb_image: NDArray[np.uint8] = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # Shape: (size, size, channels) float after resize and normalization
        transformed: NDArray = self.transforms(image=rgb_image)["image"]
        # (height, width, channels) -> (channels, height, width) for PyTorch convolutional encoders.
        item["image"] = torch.tensor(transformed).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item

    def __len__(self) -> int:
        """Return the number of aligned image-caption examples."""
        return len(self.captions)


def get_transforms() -> A.Compose:
    """Create deterministic resize and normalization transforms."""
    return A.Compose(
        [
            A.Resize(CFG.size, CFG.size),
            A.Normalize(max_pixel_value=255.0),
        ]
    )


class ImageEncoder(nn.Module):
    """Timm image encoder with its classification head removed."""

    def __init__(
        self,
        model_name: str = CFG.model_name,
        pretrained: bool = CFG.pretrained,
        trainable: bool = CFG.trainable,
    ) -> None:
        """Create the backbone and configure parameter training."""
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: Tensor) -> Tensor:
        """Encode an image batch into pooled feature vectors.

        Args:
            images: Images shaped ``(batch_size, channels, height, width)``.

        Returns:
            Features shaped ``(batch_size, image_embedding)``.
        """
        # Global average pooling removes space: (batch_size, image_embedding), 2048 for ResNet-50.
        return self.model(images)


class TextEncoder(nn.Module):
    """DistilBERT text encoder using the first token representation."""

    def __init__(
        self,
        model_name: str = CFG.text_encoder_model,
        pretrained: bool = CFG.pretrained,
        trainable: bool = CFG.trainable,
    ) -> None:
        """Create the text backbone and configure parameter training."""
        super().__init__()
        self.model = (
            DistilBertModel.from_pretrained(model_name)
            if pretrained
            else DistilBertModel(config=DistilBertConfig())
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = trainable
        self.target_token_idx = 0

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return the first-token embedding of each caption.

        Args:
            input_ids: Token IDs shaped ``(batch_size, sequence_length)``.
            attention_mask: Padding mask shaped ``(batch_size, sequence_length)``.

        Returns:
            ``[CLS]`` embeddings shaped ``(batch_size, text_embedding)``.
        """
        # last_hidden_state: (batch_size, sequence_length, text_embedding)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Keep only the [CLS] position: (batch_size, text_embedding).
        return output.last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    """Residual projection into the shared CLIP embedding space."""

    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int = CFG.projection_dim,
        dropout: float = CFG.dropout,
    ) -> None:
        """Initialize projection, residual MLP, dropout, and normalization."""
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project inputs to the normalized shared embedding space.

        Args:
            inputs: Encoder features shaped ``(batch_size, embedding_dim)``.

        Returns:
            Embeddings shaped ``(batch_size, projection_dim)``.
        """
        # (batch_size, embedding_dim) -> (batch_size, projection_dim); the rest keeps that shape.
        projected: Tensor = self.projection(inputs)
        hidden: Tensor = self.dropout(self.fc(self.gelu(projected)))
        return self.layer_norm(hidden + projected)


class CLIPModel(nn.Module):
    """Train image and text encoders with a symmetric contrastive objective."""

    def __init__(
        self,
        temperature: float = CFG.temperature,
        image_embedding: int = CFG.image_embedding,
        text_embedding: int = CFG.text_embedding,
    ) -> None:
        """Initialize encoders, projection heads, and temperature."""
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_embedding)
        self.text_projection = ProjectionHead(text_embedding)
        self.temperature = temperature

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Calculate symmetric soft-target contrastive loss for a batch.

        Args:
            batch: Mapping with ``image`` shaped ``(batch_size, channels, size, size)``
                and ``input_ids`` / ``attention_mask`` shaped ``(batch_size, sequence_length)``.

        Returns:
            Scalar mean contrastive loss.
        """
        # Shape: (batch_size, image_embedding)
        image_features: Tensor = self.image_encoder(batch["image"])
        # Shape: (batch_size, text_embedding)
        text_features: Tensor = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # Both: (batch_size, projection_dim)
        image_embeddings: Tensor = self.image_projection(image_features)
        text_embeddings: Tensor = self.text_projection(text_features)

        # (batch_size, projection_dim) @ (projection_dim, batch_size) -> (batch_size, batch_size);
        # rows are texts, columns are images, the diagonal holds the true pairs.
        logits: Tensor = (text_embeddings @ image_embeddings.T) / self.temperature
        # Shape: (batch_size, batch_size)
        image_similarity: Tensor = image_embeddings @ image_embeddings.T
        text_similarity: Tensor = text_embeddings @ text_embeddings.T
        # Soft targets, (batch_size, batch_size), so near-duplicate pairs are not punished.
        targets: Tensor = nn.functional.softmax(
            (image_similarity + text_similarity) / 2 * self.temperature,
            dim=-1,
        )
        # Per-example losses, each (batch_size,); ``.T`` swaps the text and image roles.
        text_loss: Tensor = cross_entropy(logits, targets, reduction="none")
        image_loss: Tensor = cross_entropy(logits.T, targets.T, reduction="none")
        return ((image_loss + text_loss) / 2.0).mean()


def cross_entropy(
    predictions: Tensor,
    targets: Tensor,
    reduction: Literal["none", "mean"] = "none",
) -> Tensor:
    """Calculate cross-entropy against soft target distributions.

    Args:
        predictions: Unnormalized logits shaped ``(batch, classes)``.
        targets: Soft target distributions with the same shape.
        reduction: Return per-example loss or its mean.

    Returns:
        Per-example or scalar cross-entropy loss.
    """
    # (batch, classes) -> (batch,) after summing over classes.
    losses: Tensor = (-targets * nn.functional.log_softmax(predictions, dim=-1)).sum(dim=1)
    return losses if reduction == "none" else losses.mean()
