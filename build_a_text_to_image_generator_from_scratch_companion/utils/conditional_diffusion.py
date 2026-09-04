"""Class-conditioned U-Net and DDPM training utilities."""

__all__ = [
    "DEFAULT_DEVICE",
    "device",
    "Unet",
    "EmbedLayer",
    "ResidualConvBlock",
    "UnetDown",
    "UnetUp",
    "ConditionalUNet",
    "noise_scheduler",
    "DDPM",
    "sample",
]

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

DEFAULT_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device: str = DEFAULT_DEVICE.type


class EmbedLayer(nn.Module):
    """Two-layer MLP for time or class-conditioning vectors."""

    def __init__(self, input_dim: int, emb_dim: int) -> None:
        """Initialize the embedding MLP."""
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Flatten inputs to ``(-1, input_dim)`` and embed them.

        Args:
            inputs: Conditioning values with ``batch_size * input_dim`` elements, for
                example ``(batch_size,)``, ``(batch_size, 1, 1, 1)``, or
                ``(batch_size, input_dim)``.

        Returns:
            Embeddings shaped ``(batch_size, emb_dim)``.
        """
        # (batch_size, ...) -> (batch_size, input_dim) -> (batch_size, emb_dim)
        return self.model(inputs.view(-1, self.input_dim))


class ResidualConvBlock(nn.Module):
    """Two convolution blocks with an optional scaled residual path."""

    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        """Initialize convolutions and residual behavior."""
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply both convolutions and the configured residual path.

        Args:
            inputs: Feature maps shaped ``(batch_size, in_channels, height, width)``.

        Returns:
            Feature maps shaped ``(batch_size, out_channels, height, width)``.
        """
        # Both convolutions keep spatial size: (batch_size, out_channels, height, width).
        first: Tensor = self.conv1(inputs)
        second: Tensor = self.conv2(first)
        if not self.is_res:
            return second
        residual: Tensor = inputs if self.same_channels else first
        return (residual + second) / 1.414


class UnetDown(nn.Module):
    """Residual convolution followed by two-times spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the downsampling block."""
        super().__init__()
        self.model = nn.Sequential(ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, inputs: Tensor) -> Tensor:
        """Downsample a feature map.

        Args:
            inputs: Feature maps shaped ``(batch_size, in_channels, height, width)``.

        Returns:
            Feature maps shaped ``(batch_size, out_channels, height / 2, width / 2)``.
        """
        # (batch_size, in_channels, height, width) -> (batch_size, out_channels, height / 2, width / 2)
        return self.model(inputs)


class UnetUp(nn.Module):
    """Skip concatenation followed by transposed-convolution upsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the upsampling block."""
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        """Concatenate a skip feature map and upsample.

        Args:
            inputs: Feature maps shaped ``(batch_size, channels_a, height, width)``.
            skip: Skip maps shaped ``(batch_size, channels_b, height, width)`` with
                ``channels_a + channels_b == in_channels``.

        Returns:
            Feature maps shaped ``(batch_size, out_channels, 2 * height, 2 * width)``.
        """
        # Concatenate along channels to (batch_size, in_channels, height, width), then the
        # transposed convolution doubles space: (batch_size, out_channels, 2 * height, 2 * width).
        return self.model(torch.cat((inputs, skip), dim=1))


class ConditionalUNet(nn.Module):
    """Small class-conditioned U-Net used for classifier-free guidance."""

    def __init__(self, in_channels: int, n_feat: int = 256, n_classes: int = 10) -> None:
        """Initialize encoder, conditioning, and decoder blocks."""
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.timeembed1 = EmbedLayer(1, 2 * n_feat)
        self.timeembed2 = EmbedLayer(1, n_feat)
        self.contextembed1 = EmbedLayer(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedLayer(n_classes, n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
        )

    def forward(
        self,
        inputs: Tensor,
        classes: Tensor,
        timesteps: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        """Predict noise from images, classes, times, and context masks.

        Args:
            inputs: Noisy images shaped ``(batch_size, in_channels, 28, 28)``.
            classes: Integer class labels shaped ``(batch_size,)``.
            timesteps: Timesteps normalized to ``[0, 1]``, shaped ``(batch_size,)`` or
                ``(batch_size, 1, 1, 1)``.
            context_mask: Mask shaped ``(batch_size,)``; ``1`` drops the class label.

        Returns:
            Predicted noise shaped ``(batch_size, in_channels, 28, 28)``.
        """
        # Spatial sizes below assume 28x28 inputs: the 7x7 pooling and 7x7 transposed
        # convolution hard-code that resolution.
        # (batch_size, in_channels, 28, 28) -> (batch_size, n_feat, 28, 28)
        initial: Tensor = self.init_conv(inputs)
        # Shape: (batch_size, n_feat, 14, 14)
        down1: Tensor = self.down1(initial)
        # Shape: (batch_size, 2 * n_feat, 7, 7)
        down2: Tensor = self.down2(down1)
        # AvgPool2d(7) collapses the 7x7 map: (batch_size, 2 * n_feat, 1, 1)
        hidden: Tensor = self.to_vec(down2)

        # (batch_size,) -> (batch_size, n_classes)
        one_hot_classes: Tensor = nn.functional.one_hot(classes, num_classes=self.n_classes).to(
            dtype=torch.float32
        )
        # (batch_size,) -> (batch_size, 1) -> (batch_size, n_classes)
        expanded_mask: Tensor = context_mask[:, None].repeat(1, self.n_classes)
        # Rows with mask == 1 become all zeros (context dropped); shape stays (batch_size, n_classes).
        masked_classes: Tensor = one_hot_classes * (-(1 - expanded_mask))

        # Each embedding is reshaped to (batch_size, features, 1, 1) to broadcast over space.
        class_embedding1: Tensor = self.contextembed1(masked_classes).view(
            -1, self.n_feat * 2, 1, 1
        )
        time_embedding1: Tensor = self.timeembed1(timesteps).view(-1, self.n_feat * 2, 1, 1)
        class_embedding2: Tensor = self.contextembed2(masked_classes).view(-1, self.n_feat, 1, 1)
        time_embedding2: Tensor = self.timeembed2(timesteps).view(-1, self.n_feat, 1, 1)

        # (batch_size, 2 * n_feat, 1, 1) -> (batch_size, 2 * n_feat, 7, 7)
        up1: Tensor = self.up0(hidden)
        # Concatenate with down2 -> (batch_size, 4 * n_feat, 7, 7); upsample -> (batch_size, n_feat, 14, 14)
        up2: Tensor = self.up1(class_embedding1 * up1 + time_embedding1, down2)
        # Concatenate with down1 -> (batch_size, 2 * n_feat, 14, 14); upsample -> (batch_size, n_feat, 28, 28)
        up3: Tensor = self.up2(class_embedding2 * up2 + time_embedding2, down1)
        # (batch_size, 2 * n_feat, 28, 28) -> (batch_size, in_channels, 28, 28)
        return self.out(torch.cat((up3, initial), dim=1))


# Backward-compatible class name from the upstream notebooks.
Unet = ConditionalUNet


def noise_scheduler(num_timesteps: int) -> dict[str, Tensor]:
    """Precompute linear DDPM noise-schedule coefficients.

    Args:
        num_timesteps: Number of forward diffusion steps.

    Returns:
        Named one-dimensional schedule tensors of length
        ``num_timesteps + 1``.
    """
    beta_start, beta_end = 0.0001, 0.02
    # Every schedule tensor below is (num_timesteps + 1,), indexed directly by timestep.
    beta: Tensor = (beta_end - beta_start) * torch.arange(
        0, num_timesteps + 1, dtype=torch.float32
    ) / num_timesteps + beta_start
    alpha: Tensor = 1 - beta
    alpha_bar: Tensor = torch.cumsum(torch.log(alpha), dim=0).exp()
    sqrt_one_minus_alpha_bar: Tensor = torch.sqrt(1 - alpha_bar)
    return {
        "alpha_t": alpha,
        "oneover_sqrta": 1 / torch.sqrt(alpha),
        "sqrt_beta_t": torch.sqrt(beta),
        "alphabar_t": alpha_bar,
        "sqrtab": torch.sqrt(alpha_bar),
        "sqrtmab": sqrt_one_minus_alpha_bar,
        "mab_over_sqrtmab": (1 - alpha) / sqrt_one_minus_alpha_bar,
    }


class DDPM(nn.Module):
    """Training wrapper for class-conditioned denoising diffusion."""

    def __init__(
        self,
        model: nn.Module,
        n_T: int,
        device: torch.device | str = DEFAULT_DEVICE,
        drop_prob: float = 0.1,
    ) -> None:
        """Initialize the model, schedule buffers, and MSE objective."""
        super().__init__()
        self.model = model.to(device)
        for name, values in noise_scheduler(n_T).items():
            self.register_buffer(name, values)
        self.n_T = n_T
        self.device = torch.device(device)
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, images: Tensor, classes: Tensor) -> Tensor:
        """Sample noisy training states and return noise-prediction loss.

        Args:
            images: Clean images shaped ``(batch_size, channels, height, width)``.
            classes: Integer class labels shaped ``(batch_size,)``.

        Returns:
            Scalar mean-squared error between true and predicted noise.
        """
        # Shape: (batch_size,), integer timesteps in [1, n_T]
        timesteps: Tensor = torch.randint(1, self.n_T + 1, (images.shape[0],), device=self.device)
        # Shape: (batch_size, channels, height, width)
        noise: Tensor = torch.randn_like(images)
        # Index the schedule per sample and broadcast: (batch_size,) -> (batch_size, 1, 1, 1).
        noisy_images: Tensor = (
            self.sqrtab[timesteps, None, None, None] * images
            + self.sqrtmab[timesteps, None, None, None] * noise
        )
        # Shape: (batch_size,); 1 drops the class label for classifier-free guidance.
        context_mask: Tensor = torch.bernoulli(torch.zeros_like(classes) + self.drop_prob).to(
            self.device
        )
        # Shape: (batch_size, channels, height, width); timesteps are normalized to [0, 1].
        predicted_noise: Tensor = self.model(
            noisy_images, classes, timesteps / self.n_T, context_mask
        )
        return self.loss_mse(noise, predicted_noise)


@torch.no_grad()
def sample(
    ddpm: DDPM,
    model: nn.Module,
    n_sample: int,
    size: Sequence[int],
    device: torch.device | str,
    guide_w: float = 0.0,
    step_size: int = 1,
) -> tuple[Tensor, NDArray[np.floating]]:
    """Sample images with classifier-free guidance.

    Args:
        ddpm: DDPM wrapper containing schedule buffers.
        model: Conditional noise-prediction model.
        n_sample: Number of images to generate. The upstream class schedule
            expects a multiple of ten.
        size: Per-image ``(channels, height, width)`` shape.
        device: Sampling device.
        guide_w: Classifier-free guidance strength.
        step_size: Reverse-process timestep stride.

    Returns:
        Final images shaped ``(n_sample, channels, height, width)`` and NumPy snapshots
        shaped ``(num_snapshots, n_sample, channels, height, width)``.
    """
    # Shape: (n_sample, channels, height, width), pure Gaussian noise.
    current: Tensor = torch.randn(n_sample, *size, device=device)
    # (10,) -> (n_sample,): cycle through the ten classes.
    classes: Tensor = torch.arange(0, 10, device=device)
    classes = classes.repeat(int(n_sample / classes.shape[0]))
    context_mask: Tensor = torch.zeros_like(classes, device=device)
    # Double the batch to (2 * n_sample,): first half conditional, second half unconditional.
    classes = classes.repeat(2)
    context_mask = context_mask.repeat(2)
    context_mask[n_sample:] = 1.0

    snapshots: list[NDArray[np.floating]] = []
    for timestep in range(ddpm.n_T, 0, -step_size):
        # Shape: (n_sample, 1, 1, 1), the normalized timestep per sample.
        times: Tensor = torch.full(
            (n_sample, 1, 1, 1),
            timestep / ddpm.n_T,
            device=device,
        )
        # Shape: (2 * n_sample, channels, height, width) and (2 * n_sample, 1, 1, 1)
        doubled_images: Tensor = current.repeat(2, 1, 1, 1)
        doubled_times: Tensor = times.repeat(2, 1, 1, 1)
        noise: Tensor = (
            torch.randn(n_sample, *size, device=device)
            if timestep > 1
            else torch.zeros_like(current)
        )
        # Shape: (2 * n_sample, channels, height, width)
        predictions: Tensor = model(doubled_images, classes, doubled_times, context_mask)
        # Each half: (n_sample, channels, height, width)
        conditional = predictions[:n_sample]
        unconditional = predictions[n_sample:]
        guided: Tensor = (1 + guide_w) * conditional - guide_w * unconditional
        current = (
            ddpm.oneover_sqrta[timestep] * (current - guided * ddpm.mab_over_sqrtmab[timestep])
            + ddpm.sqrt_beta_t[timestep] * noise
        )
        if timestep % 20 == 0 or timestep == ddpm.n_T or timestep < 8:
            snapshots.append(current.detach().cpu().numpy())
    # Snapshots stack to (num_snapshots, n_sample, channels, height, width).
    return current, np.array(snapshots)
