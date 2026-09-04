"""Unconditional U-Net, DDIM scheduling, data, and image utilities."""

__all__ = [
    "resolution",
    "augmentations",
    "TabularImageData",
    "transforms",
    "CustomDataset",
    "plot_losses",
    "save_images",
    "normalize_to_neg_one_to_one",
    "unnormalize_to_zero_to_one",
    "numpy_to_pil",
    "match_shape",
    "clip",
    "LayerNorm",
    "Attention",
    "get_downsample_layer",
    "get_attn_layer",
    "get_upsample_layer",
    "sinusoidal_embedding",
    "Residual",
    "PreNorm",
    "ResidualBlock",
    "UNet",
    "cosine_beta_schedule",
    "DDIMScheduler",
]

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision import utils as vision_utils
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm import tqdm

resolution: int = 64
augmentations = Compose(
    [
        Resize(resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(resolution),
        RandomHorizontalFlip(),
        ToTensor(),
    ]
)


class TabularImageData(Protocol):
    """Minimal table interface required by :class:`CustomDataset`."""

    def iterrows(self) -> Iterable[tuple[object, Mapping[str, object]]]:
        """Yield row identifiers and row mappings."""


def transforms(examples: Mapping[str, Sequence[Image.Image]]) -> dict[str, list[Tensor]]:
    """Apply the default diffusion augmentations to an image batch.

    Args:
        examples: Mapping containing an ``image`` sequence of PIL images.

    Returns:
        A mapping whose ``input`` entry lists tensors shaped
        ``(channels, resolution, resolution)``.
    """
    images: list[Tensor] = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}


class CustomDataset(Dataset):
    """Load image paths from a table and apply a tensor transform.

    Args:
        data_df: Table whose rows contain an ``image_path`` field.
        transforms: Callable applied to each RGB PIL image.
    """

    def __init__(
        self,
        data_df: TabularImageData,
        transforms: Callable[[Image.Image], Tensor],
    ) -> None:
        self.image_paths: list[str] = [str(row["image_path"]) for _, row in data_df.iterrows()]
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the number of image paths."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Load and transform one RGB image.

        Args:
            index: Example index.

        Returns:
            Mapping with ``input`` shaped ``(channels, resolution, resolution)``.
        """
        with Image.open(self.image_paths[index]) as source_image:
            # Shape: (channels, resolution, resolution) with the default augmentations.
            image: Tensor = self.transforms(source_image.convert("RGB"))
        return {"input": image}


def plot_losses(losses: Sequence[float], out_dir: str | Path) -> None:
    """Save a training-loss line chart.

    Args:
        losses: Loss values in training order.
        out_dir: Directory in which ``losses.png`` is written.
    """
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.plot(losses, label="train")
    plt.legend()
    plt.savefig(output_directory / "losses.png")
    plt.clf()


def save_images(
    generated_images: Mapping[str, Tensor | NDArray[np.floating]],
    epoch: int,
    batch_size: int,
) -> None:
    """Save individual generated images and a PyTorch image grid.

    Args:
        generated_images: Mapping with NumPy ``sample`` images shaped
            ``(batch_size, height, width, channels)`` in ``[0, 1]`` and a tensor
            ``sample_pt`` batch shaped ``(batch_size, channels, height, width)``.
        epoch: Epoch number used for output names.
        batch_size: Batch size used to choose the grid width.

    Raises:
        TypeError: If either expected mapping value has the wrong type.
    """
    samples = generated_images["sample"]
    sample_tensor = generated_images["sample_pt"]
    if not isinstance(samples, np.ndarray) or not isinstance(sample_tensor, Tensor):
        raise TypeError("sample must be an ndarray and sample_pt must be a tensor")

    # Shape: (batch_size, height, width, channels) uint8
    images: NDArray[np.uint8] = (samples * 255).round().astype("uint8")
    output_directory = Path(str(epoch))
    output_directory.mkdir()
    for index, image in enumerate(images):
        Image.fromarray(image).save(output_directory / f"{epoch}_{index}.jpeg")
    vision_utils.save_image(
        sample_tensor,
        output_directory / f"{epoch}_grid.jpeg",
        nrow=batch_size // 4,
    )


def normalize_to_neg_one_to_one(image: Tensor) -> Tensor:
    """Map values from ``[0, 1]`` to ``[-1, 1]`` element-wise, preserving shape."""
    return image * 2 - 1


def unnormalize_to_zero_to_one(tensor: Tensor) -> Tensor:
    """Map values from ``[-1, 1]`` to ``[0, 1]`` element-wise, preserving shape."""
    return (tensor + 1) * 0.5


def _to_numpy_images(images: Tensor) -> NDArray[np.floating]:
    """Convert model-space images to channels-last NumPy arrays in ``[0, 1]``.

    Args:
        images: Images shaped ``(batch_size, channels, height, width)`` in ``[-1, 1]``.

    Returns:
        Arrays shaped ``(batch_size, height, width, channels)`` in ``[0, 1]``.
    """
    return unnormalize_to_zero_to_one(images).cpu().permute(0, 2, 3, 1).numpy()


def numpy_to_pil(images: NDArray[np.floating]) -> list[Image.Image]:
    """Convert one image or a batch of ``[0, 1]`` arrays to PIL images.

    Args:
        images: One ``(height, width, channels)`` image or a batch shaped
            ``(batch_size, height, width, channels)``.

    Returns:
        Converted PIL images.
    """
    # Ensure a batch axis: (batch_size, height, width, channels).
    batched: NDArray[np.floating] = images[None, ...] if images.ndim == 3 else images
    uint8_images: NDArray[np.uint8] = (batched * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in uint8_images]


@overload
def match_shape(values: Tensor, broadcast_array: Tensor, tensor_format: str = "pt") -> Tensor: ...


@overload
def match_shape(
    values: NDArray, broadcast_array: NDArray, tensor_format: str = "np"
) -> NDArray: ...


def match_shape(
    values: Tensor | NDArray,
    broadcast_array: Tensor | NDArray,
    tensor_format: str = "pt",
) -> Tensor | NDArray:
    """Append singleton dimensions until values broadcast over an array.

    Args:
        values: Values indexed by batch or timestep.
        broadcast_array: Target whose rank determines the result rank.
        tensor_format: ``"pt"`` moves tensor values to the target device;
            other values leave the device unchanged.

    Returns:
        Reshaped values with the same rank as ``broadcast_array``.
    """
    # (batch_size,) -> (batch_size, 1, 1, 1) for a 4-D image target, so values broadcast per sample.
    shaped = values.flatten()
    while shaped.ndim < broadcast_array.ndim:
        shaped = shaped[..., None]
    if tensor_format == "pt" and isinstance(shaped, Tensor):
        if not isinstance(broadcast_array, Tensor):
            raise TypeError("PyTorch format requires a tensor broadcast target")
        shaped = shaped.to(broadcast_array.device)
    return shaped


@overload
def clip(
    tensor: Tensor,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tensor: ...


@overload
def clip(
    tensor: NDArray,
    min_value: float | None = None,
    max_value: float | None = None,
) -> NDArray: ...


def clip(
    tensor: Tensor | NDArray,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Tensor | NDArray:
    """Clip a NumPy array or PyTorch tensor element-wise, preserving shape.

    Raises:
        ValueError: If ``tensor`` is neither a NumPy array nor a tensor.
    """
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    if isinstance(tensor, Tensor):
        return torch.clamp(tensor, min_value, max_value)
    raise ValueError(
        f"Tensor format is not valid; expected a NumPy array or PyTorch tensor, got {type(tensor)}."
    )


class LayerNorm(nn.Module):
    """Channel-wise normalization for image feature maps."""

    def __init__(self, dim: int) -> None:
        """Initialize the learned channel scale."""
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize each pixel across its channels.

        Args:
            inputs: Feature maps shaped ``(batch_size, channels, height, width)``.

        Returns:
            Normalized maps shaped ``(batch_size, channels, height, width)``.
        """
        eps: float = 1e-5 if inputs.dtype == torch.float32 else 1e-3
        # Statistics over channels: (batch_size, 1, height, width); ``g`` is (1, dim, 1, 1).
        variance: Tensor = torch.var(inputs, dim=1, unbiased=False, keepdim=True)
        mean: Tensor = torch.mean(inputs, dim=1, keepdim=True)
        return (inputs - mean) * (variance + eps).rsqrt() * self.g


class Attention(nn.Module):
    """Multi-head self-attention over spatial image positions."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        """Initialize convolutional attention projections."""
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim: int = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Attend over flattened spatial positions and restore image shape.

        Args:
            inputs: Feature maps shaped ``(batch_size, dim, height, width)``.

        Returns:
            Attended maps shaped ``(batch_size, dim, height, width)``.
        """
        _, _, height, width = inputs.shape
        # (batch_size, dim, height, width) -> (batch_size, 3 * hidden_dim, height, width),
        # chunked along channels into three (batch_size, hidden_dim, height, width) tensors.
        query, key, value = self.to_qkv(inputs).chunk(3, dim=1)
        # Split heads and flatten space:
        # (batch_size, heads * dim_head, height, width) -> (batch_size, heads, dim_head, height * width)
        query, key, value = [
            rearrange(tensor, "b (heads c) x y -> b heads c (x y)", heads=self.heads)
            for tensor in (query, key, value)
        ]
        query = query * self.scale
        # Shape: (batch_size, heads, positions, positions) with positions = height * width
        similarity: Tensor = einsum(query, key, "b h d i, b h d j -> b h i j")
        probabilities: Tensor = similarity.softmax(dim=-1)
        # (batch_size, heads, positions, positions) x (batch_size, heads, dim_head, positions)
        # -> (batch_size, heads, positions, dim_head)
        attended: Tensor = einsum(probabilities, value, "b h i j, b h d j -> b h i d")
        # (batch_size, heads, positions, dim_head) -> (batch_size, heads * dim_head, height, width)
        restored: Tensor = rearrange(
            attended,
            "b h (x y) d -> b (h d) x y",
            x=height,
            y=width,
        )
        # (batch_size, hidden_dim, height, width) -> (batch_size, dim, height, width)
        return self.to_out(restored)


def get_downsample_layer(in_dim: int, hidden_dim: int, is_last: bool) -> nn.Module:
    """Create the configured U-Net downsampling operation.

    Returns:
        Module mapping ``(batch_size, in_dim, height, width)`` to
        ``(batch_size, hidden_dim, height / 2, width / 2)``, or keeping the spatial
        size when ``is_last``.
    """
    if is_last:
        # Keeps spatial size: (batch_size, in_dim, height, width) -> (batch_size, hidden_dim, height, width)
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
    # Space-to-depth: (batch_size, in_dim, height, width) -> (batch_size, 4 * in_dim, height / 2, width / 2),
    # then a 1x1 convolution to (batch_size, hidden_dim, height / 2, width / 2).
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(in_dim * 4, hidden_dim, 1),
    )


def get_attn_layer(in_dim: int, is_last: bool) -> nn.Module:
    """Create spatial attention only for the innermost resolution.

    Returns:
        Shape-preserving module over ``(batch_size, in_dim, height, width)``.
    """
    if is_last:
        return Residual(PreNorm(in_dim, Attention(in_dim)))
    return nn.Identity()


def get_upsample_layer(in_dim: int, hidden_dim: int, is_last: bool) -> nn.Module:
    """Create the configured U-Net upsampling operation.

    Returns:
        Module mapping ``(batch_size, in_dim, height, width)`` to
        ``(batch_size, hidden_dim, 2 * height, 2 * width)``, or keeping the spatial
        size when ``is_last``.
    """
    if is_last:
        # Keeps spatial size: (batch_size, in_dim, height, width) -> (batch_size, hidden_dim, height, width)
        return nn.Conv2d(in_dim, hidden_dim, 3, padding=1)
    # (batch_size, in_dim, height, width) -> (batch_size, in_dim, 2 * height, 2 * width)
    # -> (batch_size, hidden_dim, 2 * height, 2 * width)
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
    )


def sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Create sinusoidal diffusion-timestep embeddings.

    Args:
        timesteps: Integer timesteps shaped ``(batch_size,)``.
        dim: Embedding size.

    Returns:
        Embeddings shaped ``(batch_size, dim)``.
    """
    half_dim: int = dim // 2
    exponent: Tensor = (
        -math.log(10_000) * torch.arange(0, half_dim, dtype=torch.float32) / (half_dim - 1.0)
    )
    # Shape: (half_dim,)
    frequencies: Tensor = torch.exp(exponent).to(timesteps.device)
    # (batch_size, 1) * (1, half_dim) -> (batch_size, half_dim)
    phases: Tensor = timesteps[:, None].float() * frequencies[None, :]
    # Concatenate sin and cos halves: (batch_size, dim).
    return torch.cat([phases.sin(), phases.cos()], dim=-1)


class Residual(nn.Module):
    """Add a module's output to its input."""

    def __init__(self, function: nn.Module) -> None:
        """Store the residual branch module."""
        super().__init__()
        self.fn = function

    def forward(self, inputs: Tensor, *args: Tensor, **kwargs: Tensor) -> Tensor:
        """Apply the branch and add the unchanged input.

        Args:
            inputs: Tensor of any shape; the branch must preserve it.

        Returns:
            Residual sum with the same shape as ``inputs``.
        """
        return self.fn(inputs, *args, **kwargs) + inputs


class PreNorm(nn.Module):
    """Apply channel-wise normalization before another module."""

    def __init__(self, dim: int, function: nn.Module) -> None:
        """Initialize normalization and wrapped module."""
        super().__init__()
        self.fn = function
        self.norm = LayerNorm(dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize inputs before applying the wrapped module.

        Args:
            inputs: Feature maps shaped ``(batch_size, dim, height, width)``.

        Returns:
            Output of the wrapped module, shaped like ``inputs``.
        """
        return self.fn(self.norm(inputs))


class ResidualBlock(nn.Module):
    """Time-conditioned residual convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 8,
    ) -> None:
        """Initialize residual, time, convolution, and normalization paths."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_proj = nn.Sequential(nn.SiLU(), nn.Linear(temb_channels, out_channels))
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.nonlinearity = nn.SiLU()

    def forward(self, inputs: Tensor, time_embedding: Tensor) -> Tensor:
        """Apply convolutions with a broadcast time embedding.

        Args:
            inputs: Feature maps shaped ``(batch_size, in_channels, height, width)``.
            time_embedding: Time features shaped ``(batch_size, temb_channels)``.

        Returns:
            Feature maps shaped ``(batch_size, out_channels, height, width)``.
        """
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height, width)
        residual: Tensor = self.residual_conv(inputs)
        # Shape: (batch_size, out_channels, height, width)
        hidden: Tensor = self.nonlinearity(self.norm1(self.conv1(inputs)))
        # (batch_size, temb_channels) -> (batch_size, out_channels)
        projected_time: Tensor = self.time_emb_proj(self.nonlinearity(time_embedding))
        # Broadcast the time vector over space as (batch_size, out_channels, 1, 1).
        hidden = hidden + projected_time[:, :, None, None]
        hidden = self.nonlinearity(self.norm2(self.conv2(hidden)))
        return hidden + residual


class UNet(nn.Module):
    """Time-conditioned denoising U-Net with bottleneck attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Sequence[int] = (128, 256, 512, 1024),
        image_size: int = 64,
    ) -> None:
        """Initialize downsampling, bottleneck, and upsampling paths."""
        super().__init__()
        dimensions = list(hidden_dims)
        self.sample_size = image_size
        self.in_channels = in_channels
        self.hidden_dims = dimensions
        timestep_input_dim: int = dimensions[0]
        time_embed_dim: int = timestep_input_dim * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, dimensions[0], kernel_size=3, stride=1, padding=1)

        down_blocks: list[nn.ModuleList] = []
        in_dim: int = dimensions[0]
        for index, hidden_dim in enumerate(dimensions[1:]):
            is_last: bool = index >= len(dimensions) - 2
            down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_dim, in_dim, time_embed_dim),
                        ResidualBlock(in_dim, in_dim, time_embed_dim),
                        get_attn_layer(in_dim, is_last),
                        get_downsample_layer(in_dim, hidden_dim, is_last),
                    ]
                )
            )
            in_dim = hidden_dim
        self.down_blocks = nn.ModuleList(down_blocks)

        middle_dim: int = dimensions[-1]
        self.mid_block1 = ResidualBlock(middle_dim, middle_dim, time_embed_dim)
        self.mid_attn = Residual(PreNorm(middle_dim, Attention(middle_dim)))
        self.mid_block2 = ResidualBlock(middle_dim, middle_dim, time_embed_dim)

        up_blocks: list[nn.ModuleList] = []
        in_dim = middle_dim
        for index, hidden_dim in enumerate(reversed(dimensions[:-1])):
            is_last = index >= len(dimensions) - 2
            up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                        ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim),
                        get_attn_layer(in_dim, is_last),
                        get_upsample_layer(in_dim, hidden_dim, is_last),
                    ]
                )
            )
            in_dim = hidden_dim
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_block = ResidualBlock(dimensions[0] * 2, dimensions[0], time_embed_dim)
        self.conv_out = nn.Conv2d(dimensions[0], 3, kernel_size=1)

    def forward(self, sample: Tensor, timesteps: Tensor | int) -> dict[str, Tensor]:
        """Predict image noise for one or more diffusion timesteps.

        Args:
            sample: Noisy images shaped ``(batch_size, in_channels, image_size, image_size)``.
            timesteps: One integer timestep for the whole batch, or a tensor
                broadcastable to ``(batch_size,)``.

        Returns:
            Mapping with ``sample`` holding predicted noise shaped
            ``(batch_size, 3, image_size, image_size)``.
        """
        if not isinstance(timesteps, Tensor):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # Shape: (batch_size,), one timestep per sample.
        expanded_timesteps: Tensor = torch.flatten(timesteps).broadcast_to(sample.shape[0])
        # (batch_size,) -> (batch_size, hidden_dims[0]) -> (batch_size, time_embed_dim)
        time_embedding: Tensor = self.time_embedding(
            sinusoidal_embedding(expanded_timesteps, self.hidden_dims[0])
        )
        # (batch_size, in_channels, image_size, image_size) -> (batch_size, hidden_dims[0], image_size, image_size)
        hidden: Tensor = self.init_conv(sample)
        initial_residual: Tensor = hidden.clone()
        skips: list[Tensor] = []

        # Each stage widens channels to the next hidden_dim and halves height and width
        # (the last stage keeps spatial size). Two skips are saved per stage.
        for block1, block2, attention_layer, downsample in self.down_blocks:
            hidden = block1(hidden, time_embedding)
            skips.append(hidden)
            hidden = block2(hidden, time_embedding)
            hidden = attention_layer(hidden)
            skips.append(hidden)
            hidden = downsample(hidden)

        hidden = self.mid_block1(hidden, time_embedding)
        hidden = self.mid_attn(hidden)
        hidden = self.mid_block2(hidden, time_embedding)

        # Mirror the down path: concatenate the matching skip along channels,
        # (batch_size, in_dim + skip_dim, height, width), then double height and width.
        for block1, block2, attention_layer, upsample in self.up_blocks:
            hidden = block1(torch.cat((hidden, skips.pop()), dim=1), time_embedding)
            hidden = block2(torch.cat((hidden, skips.pop()), dim=1), time_embedding)
            hidden = attention_layer(hidden)
            hidden = upsample(hidden)

        # (batch_size, 2 * hidden_dims[0], image_size, image_size) -> (batch_size, hidden_dims[0], ...)
        hidden = self.out_block(torch.cat((hidden, initial_residual), dim=1), time_embedding)
        # (batch_size, hidden_dims[0], image_size, image_size) -> (batch_size, 3, image_size, image_size)
        return {"sample": self.conv_out(hidden)}


def cosine_beta_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    s: float = 0.008,
) -> Tensor:
    """Create the clipped cosine beta schedule from Improved DDPM.

    Returns:
        Betas shaped ``(timesteps,)`` clipped to ``[beta_start, beta_end]``.
    """
    steps: int = timesteps + 1
    # Shape: (timesteps + 1,)
    positions: Tensor = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alpha_bar: Tensor = torch.cos(((positions / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    # Adjacent ratios: (timesteps,)
    betas: Tensor = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(betas, beta_start, beta_end)


class DDIMScheduler:
    """Denoising Diffusion Implicit Model scheduler and sampler."""

    def __init__(
        self,
        num_train_timesteps: int = 1_000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
    ) -> None:
        """Initialize training noise coefficients."""
        if beta_schedule == "linear":
            betas: NDArray[np.float32] = np.linspace(
                beta_start,
                beta_end,
                num_train_timesteps,
                dtype=np.float32,
            )
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps, beta_start, beta_end).numpy()
        else:
            raise NotImplementedError(
                f"{beta_schedule} is not implemented for {self.__class__.__name__}"
            )

        # All schedule arrays are (num_train_timesteps,), indexed by training timestep.
        self.betas = betas
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.alphas: NDArray[np.float32] = 1.0 - self.betas
        self.alphas_cumprod: NDArray[np.float32] = np.cumprod(self.alphas, axis=0)
        self.final_alpha_cumprod: np.float64 | np.float32 = (
            np.float64(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.num_inference_steps: int | None = None
        self.timesteps: NDArray[np.int_] = np.arange(num_train_timesteps)[::-1].copy()

    def _get_variance(self, timestep: int, prev_timestep: int) -> np.floating:
        """Return DDIM variance between two selected timesteps."""
        alpha_current = self.alphas_cumprod[timestep]
        alpha_previous = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_current = 1 - alpha_current
        beta_previous = 1 - alpha_previous
        return (beta_previous / beta_current) * (1 - alpha_current / alpha_previous)

    def set_timesteps(self, num_inference_steps: int, offset: int = 0) -> None:
        """Select evenly spaced reverse-process timesteps."""
        self.num_inference_steps = num_inference_steps
        stride: int = self.num_train_timesteps // num_inference_steps
        # Shape: (num_inference_steps,), descending from the noisiest timestep.
        self.timesteps = np.arange(0, self.num_train_timesteps, stride, dtype=np.int_)[::-1].copy()
        self.timesteps += offset

    def step(
        self,
        model_output: Tensor | NDArray,
        timestep: int,
        sample: Tensor | NDArray,
        eta: float = 1.0,
        use_clipped_model_output: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor | NDArray:
        """Predict the sample at the previous selected DDIM timestep.

        Args:
            model_output: Predicted noise shaped ``(batch_size, channels, height, width)``.
            timestep: Current training timestep index.
            sample: Current noisy sample shaped ``(batch_size, channels, height, width)``.
            eta: Stochasticity weight; ``0`` gives deterministic DDIM.
            use_clipped_model_output: Re-derive noise from the clipped prediction.
            generator: Optional random generator for the added noise.

        Returns:
            Previous sample shaped ``(batch_size, channels, height, width)``.

        Raises:
            RuntimeError: If :meth:`set_timesteps` has not been called.
        """
        if self.num_inference_steps is None:
            raise RuntimeError("call set_timesteps before step")
        previous_timestep: int = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_current = self.alphas_cumprod[timestep]
        alpha_previous = (
            self.alphas_cumprod[previous_timestep]
            if previous_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_current = 1 - alpha_current
        predicted_original = (sample - beta_current**0.5 * model_output) / alpha_current**0.5
        if self.clip_sample:
            predicted_original = clip(predicted_original, -1, 1)

        variance = self._get_variance(timestep, previous_timestep)
        standard_deviation = eta * variance**0.5
        if use_clipped_model_output:
            model_output = (sample - alpha_current**0.5 * predicted_original) / beta_current**0.5
        direction = (1 - alpha_previous - standard_deviation**2) ** 0.5 * model_output
        previous_sample = alpha_previous**0.5 * predicted_original + direction

        if eta > 0:
            noise = torch.randn(model_output.shape, generator=generator)
            if isinstance(model_output, Tensor):
                noise = noise.to(model_output.device)
                random_variance: Tensor | NDArray = variance**0.5 * eta * noise
            else:
                random_variance = (variance**0.5 * eta * noise).numpy()
            previous_sample = previous_sample + random_variance
        return previous_sample

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Apply forward-process noise at selected timesteps.

        Args:
            original_samples: Clean images shaped ``(batch_size, channels, height, width)``.
            noise: Gaussian noise shaped ``(batch_size, channels, height, width)``.
            timesteps: Integer timesteps shaped ``(batch_size,)``.

        Returns:
            Noisy images shaped ``(batch_size, channels, height, width)``.
        """
        # timesteps: (batch_size,) -> per-sample coefficients (batch_size,)
        cpu_timesteps: Tensor = timesteps.cpu()
        sqrt_alpha = torch.from_numpy(self.alphas_cumprod[cpu_timesteps.numpy()] ** 0.5).to(
            dtype=original_samples.dtype
        )
        # (batch_size,) -> (batch_size, 1, 1, 1) so each coefficient broadcasts over its image.
        sqrt_alpha = match_shape(sqrt_alpha, original_samples)
        sqrt_one_minus_alpha = torch.from_numpy(
            (1 - self.alphas_cumprod[cpu_timesteps.numpy()]) ** 0.5
        ).to(dtype=original_samples.dtype)
        sqrt_one_minus_alpha = match_shape(sqrt_one_minus_alpha, original_samples)
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def generate(
        self,
        model: nn.Module,
        device: torch.device | str,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        eta: float = 1.0,
        use_clipped_model_output: bool = True,
        num_inference_steps: int = 50,
    ) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
        """Generate images and retain all intermediate denoising states.

        Returns:
            Final images shaped ``(batch_size, sample_size, sample_size, in_channels)``
            in ``[0, 1]`` and a list of ``num_inference_steps`` snapshots with the
            same shape.
        """
        sample_size = int(model.sample_size)
        in_channels = int(model.in_channels)
        # Shape: (batch_size, in_channels, sample_size, sample_size), pure Gaussian noise.
        image: Tensor = torch.randn(
            (batch_size, in_channels, sample_size, sample_size),
            generator=generator,
            device=device,
        )
        snapshots: list[NDArray[np.floating]] = [
            _to_numpy_images(denoised)
            for denoised in self._denoise(
                model, image, num_inference_steps, eta, use_clipped_model_output, generator
            )
        ]
        return snapshots[-1].copy(), snapshots

    @torch.no_grad()
    def interpolate(
        self,
        model: nn.Module,
        a_idx: int,
        b_idx: int,
        batch_size: int = 1,
        generator: torch.Generator | None = None,
        eta: float = 1.0,
        use_clipped_model_output: bool = True,
        num_inference_steps: int = 50,
        device: torch.device | str | None = None,
    ) -> NDArray[np.floating]:
        """Generate ten spherical interpolations between two noise samples.

        Returns:
            Images shaped ``(batch_size, sample_size, sample_size, in_channels)`` in
            ``[0, 1]``; rows ``0`` to ``9`` hold the interpolations.

        Raises:
            ValueError: If ``batch_size`` is smaller than ten.
        """
        if batch_size < 10:
            raise ValueError("batch_size must be at least 10 for interpolation")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sample_size = int(model.sample_size)
        in_channels = int(model.in_channels)
        initial_noise: Tensor = torch.randn(
            (batch_size, in_channels, sample_size, sample_size),
            generator=generator,
            device=selected_device,
        )
        image: Tensor = torch.zeros_like(initial_noise)
        # Two endpoints, each (in_channels, sample_size, sample_size); rows 0-9 hold the blends.
        first, second = initial_noise[a_idx], initial_noise[b_idx]
        for index in range(10):
            angle = torch.tensor(0.5 * math.pi * (0.05 + index / 10))
            image[index] = torch.sin(angle) * first + torch.cos(angle) * second

        *_, image = self._denoise(
            model, image, num_inference_steps, eta, use_clipped_model_output, generator
        )
        return _to_numpy_images(image)

    def _denoise(
        self,
        model: nn.Module,
        image: Tensor,
        num_inference_steps: int,
        eta: float,
        use_clipped_model_output: bool,
        generator: torch.Generator | None,
    ) -> Iterator[Tensor]:
        """Run the reverse process, yielding the sample after every DDIM step.

        Args:
            model: Noise-prediction model returning ``{"sample": noise}``.
            image: Starting noise shaped ``(batch_size, channels, height, width)``.
            num_inference_steps: Number of evenly spaced reverse steps.
            eta: Stochasticity weight passed to :meth:`step`.
            use_clipped_model_output: Passed to :meth:`step`.
            generator: Optional random generator for the added noise.

        Yields:
            The current sample shaped ``(batch_size, channels, height, width)``.
        """
        self.set_timesteps(num_inference_steps)
        for timestep_value in tqdm(self.timesteps):
            timestep = int(timestep_value)
            stepped = self.step(
                model(image, timestep)["sample"],
                timestep,
                image,
                eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            )
            if not isinstance(stepped, Tensor):
                raise TypeError("tensor model output must produce a tensor sample")
            image = stepped
            yield image

    def __len__(self) -> int:
        """Return the number of training timesteps."""
        return self.num_train_timesteps
