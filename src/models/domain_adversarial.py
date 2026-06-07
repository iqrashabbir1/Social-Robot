from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models.torch_runtime import require_torch


torch = require_torch()
nn = torch.nn


class GradientReversalFunction(torch.autograd.Function):
    """Autograd function that multiplies incoming gradients by ``-alpha``.

    The forward pass is the identity function. During backpropagation the
    gradient is reversed so that the feature extractor is optimized to confuse
    the domain discriminator while still minimizing the emotion classification
    loss.
    """

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for the gradient reversal autograd function."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, inputs: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        effective_alpha = self.alpha if alpha is None else float(alpha)
        return GradientReversalFunction.apply(inputs, effective_alpha)


class DomainAdversarialNetwork(nn.Module):
    """Domain discriminator used by the DANN model.

    The requested architecture is preserved exactly at the hidden-layer level:
    ``384 -> 256 -> 128 -> 2``.
    """

    def __init__(self, feature_dim: int = 384, hidden_dims: tuple[int, int] = (256, 128), dropout: float = 0.2) -> None:
        super().__init__()
        self.grl = GradientReversalLayer()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 2),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_features = self.grl(features, alpha)
        return self.discriminator(reversed_features)


class _CNNFeatureExtractor(nn.Module):
    """CNN-small style visual encoder with a 384-D feature projection.

    This mirrors the repository's current compact visual baseline while exposing
    a stable intermediate feature representation for domain adaptation.
    """

    def __init__(self, image_size: int = 128, feature_dim: int = 384) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 4 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(images))


@dataclass(frozen=True)
class DANNOutput:
    features: torch.Tensor
    emotion_logits: torch.Tensor
    domain_logits: torch.Tensor


class DANNMultimodalEmotionModel(nn.Module):
    """Multimodal-ready DANN emotion model.

    The current repository benchmark is image-driven, so the feature extractor
    uses the compact CNN-small visual pathway. An optional auxiliary feature
    branch is kept to support later multimodal expansion without changing the
    domain-adaptation interface.
    """

    def __init__(
        self,
        num_classes: int = 5,
        image_size: int = 128,
        feature_dim: int = 384,
        auxiliary_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.feature_dim = int(feature_dim)
        self.auxiliary_dim = int(auxiliary_dim)

        self.visual_extractor = _CNNFeatureExtractor(image_size=image_size, feature_dim=feature_dim)
        self.auxiliary_encoder = None
        self.fusion_projection = None

        if self.auxiliary_dim > 0:
            self.auxiliary_encoder = nn.Sequential(
                nn.Linear(self.auxiliary_dim, 96),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.fusion_projection = nn.Sequential(
                nn.Linear(feature_dim + 96, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, self.num_classes),
        )
        self.domain_classifier = DomainAdversarialNetwork(feature_dim=feature_dim)

    def extract_features(
        self,
        images: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_features = self.visual_extractor(images)
        if self.auxiliary_encoder is None or auxiliary_features is None:
            return visual_features
        encoded_aux = self.auxiliary_encoder(auxiliary_features)
        fused = torch.cat([visual_features, encoded_aux], dim=1)
        return self.fusion_projection(fused)

    def forward(
        self,
        images: torch.Tensor,
        auxiliary_features: torch.Tensor | None = None,
        grl_alpha: float = 1.0,
    ) -> DANNOutput:
        features = self.extract_features(images, auxiliary_features=auxiliary_features)
        emotion_logits = self.emotion_classifier(features)
        domain_logits = self.domain_classifier(features, alpha=grl_alpha)
        return DANNOutput(
            features=features,
            emotion_logits=emotion_logits,
            domain_logits=domain_logits,
        )
