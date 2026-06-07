from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


try:  # pragma: no cover - optional runtime dependency
    from opacus import PrivacyEngine as OpacusPrivacyEngine
except ImportError:  # pragma: no cover - optional runtime dependency
    OpacusPrivacyEngine = None


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    steps: int
    epochs: int
    source: str


def compute_closed_form_epsilon(
    *,
    noise_multiplier: float,
    steps: int,
    delta: float,
) -> float:
    """Simplified privacy-budget estimate requested for manuscript tables.

    Notes
    -----
    This is a compact closed-form estimate intended for reporting schedules in
    tables and quick comparisons. During actual training, the accountant-based
    epsilon computed by Opacus should be treated as the authoritative value.
    """

    safe_steps = max(int(steps), 0)
    safe_delta = max(float(delta), 1e-12)
    return float(noise_multiplier * math.sqrt(safe_steps * math.log(1.0 / safe_delta)))


def compute_epsilon_for_epochs(
    *,
    epochs: int,
    steps_per_epoch: int,
    noise_multiplier: float = 1.1,
    delta: float = 1e-5,
) -> float:
    total_steps = max(int(epochs), 0) * max(int(steps_per_epoch), 0)
    return compute_closed_form_epsilon(
        noise_multiplier=noise_multiplier,
        steps=total_steps,
        delta=delta,
    )


class PrivacyEngine:
    """Thin wrapper around Opacus DP-SGD with paper-friendly reporting helpers."""

    def __init__(
        self,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        accountant: str = "rdp",
        secure_mode: bool = False,
    ) -> None:
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.delta = float(delta)
        self.accountant = accountant
        self.secure_mode = bool(secure_mode)
        self.steps = 0
        self.epochs = 0
        self._sample_rate = 0.0

        if OpacusPrivacyEngine is None:
            self._engine = None
        else:
            self._engine = OpacusPrivacyEngine(accountant=accountant, secure_mode=secure_mode)

    @property
    def is_available(self) -> bool:
        return self._engine is not None

    def require_backend(self) -> None:
        if self._engine is None:
            raise RuntimeError(
                "Opacus is required for differential privacy training. "
                "Install it with `pip install opacus` before using the DP trainer."
            )

    def make_private(
        self,
        *,
        module: Any,
        optimizer: Any,
        data_loader: Any,
        epochs: int,
        poisson_sampling: bool = True,
    ) -> tuple[Any, Any, Any]:
        self.require_backend()
        private_module, private_optimizer, private_loader = self._engine.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=poisson_sampling,
        )
        self.epochs = int(epochs)
        try:
            dataset_size = len(data_loader.dataset)
            batch_size = getattr(data_loader, "batch_size", None) or 0
            self._sample_rate = float(batch_size) / float(max(dataset_size, 1))
        except Exception:
            self._sample_rate = 0.0
        return private_module, private_optimizer, private_loader

    def step(self, step_count: int = 1) -> None:
        self.steps += max(int(step_count), 0)

    def compute_epsilon(
        self,
        delta: float | None = None,
    ) -> float:
        effective_delta = float(self.delta if delta is None else delta)
        if self._engine is not None and hasattr(self._engine, "accountant"):
            try:
                return float(self._engine.accountant.get_epsilon(delta=effective_delta))
            except Exception:
                pass
        return compute_closed_form_epsilon(
            noise_multiplier=self.noise_multiplier,
            steps=self.steps,
            delta=effective_delta,
        )

    def get_privacy_budget(self, delta: float | None = None) -> PrivacyBudget:
        effective_delta = float(self.delta if delta is None else delta)
        source = "opacus_rdp_accountant" if self._engine is not None else "closed_form_estimate"
        return PrivacyBudget(
            epsilon=self.compute_epsilon(delta=effective_delta),
            delta=effective_delta,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            steps=self.steps,
            epochs=self.epochs,
            source=source,
        )
