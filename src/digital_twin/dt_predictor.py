from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths
from src.models.torch_runtime import require_torch


torch = require_torch()
nn = torch.nn


class _LSTMStateModel(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, sequence_batch: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequence_batch)
        return self.regressor(outputs[:, -1, :])


@dataclass(frozen=True)
class PredictionConfidence:
    mean_prediction: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    std_prediction: np.ndarray


def simulate_dt_sequences(
    num_sequences: int = 256,
    sequence_length: int = 20,
    input_dim: int = 384,
    prediction_horizon: int = 10,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    latent_dim = 6
    projection = rng.normal(0.0, 0.16, size=(latent_dim, input_dim)).astype(np.float32)
    residual_projection = rng.normal(0.0, 0.035, size=(latent_dim, input_dim)).astype(np.float32)
    sequences = np.zeros((num_sequences, sequence_length, input_dim), dtype=np.float32)
    future_states = np.zeros((num_sequences, prediction_horizon, input_dim), dtype=np.float32)

    for seq_index in range(num_sequences):
        latent = rng.normal(0.0, 0.12, size=latent_dim).astype(np.float32)
        sequence_rows: list[np.ndarray] = []
        future_rows: list[np.ndarray] = []
        previous_emission = np.zeros(input_dim, dtype=np.float32)
        for step in range(sequence_length + prediction_horizon):
            phase = np.arange(latent_dim, dtype=np.float32)
            seasonal = 0.015 * np.sin(0.045 * step + phase) + 0.008 * np.cos(0.03 * step + phase / 2.0)
            latent = (0.975 * latent + seasonal + rng.normal(0.0, 0.0015, size=latent_dim)).astype(np.float32)

            structural = latent @ projection
            residual = latent @ residual_projection
            emission = (
                0.94 * previous_emission
                + 0.045 * np.tanh(structural)
                + 0.015 * residual
                + rng.normal(0.0, 0.0015, size=input_dim)
            ).astype(np.float32)
            emission = np.clip(emission, -1.0, 1.0)
            previous_emission = emission
            if step < sequence_length:
                sequence_rows.append(emission)
            else:
                future_rows.append(emission)
        sequences[seq_index] = np.stack(sequence_rows, axis=0)
        future_states[seq_index] = np.stack(future_rows, axis=0)

    return sequences, future_states


class DTPredictor:
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.input_dim = int(input_dim)
        self.device = device
        self.model = _LSTMStateModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

    def fit(
        self,
        history_sequences: np.ndarray,
        next_state_targets: np.ndarray,
        *,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> dict[str, float]:
        features = torch.tensor(history_sequences, dtype=torch.float32, device=self.device)
        targets = torch.tensor(next_state_targets, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(features, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        history_losses: list[float] = []

        self.model.train()
        for _epoch in range(epochs):
            epoch_losses: list[float] = []
            for batch_features, batch_targets in loader:
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_features)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            history_losses.append(float(np.mean(epoch_losses)))
        return {
            "final_train_loss": float(history_losses[-1]),
            "mean_train_loss": float(np.mean(history_losses)),
        }

    def predict_next_state(
        self,
        history: np.ndarray,
        horizon_seconds: int = 10,
        step_seconds: float = 1.0,
    ) -> np.ndarray:
        steps = max(1, int(round(horizon_seconds / max(step_seconds, 1e-6))))
        current_history = np.asarray(history, dtype=np.float32)
        predictions: list[np.ndarray] = []

        self.model.eval()
        for _step in range(steps):
            history_tensor = torch.tensor(current_history[None, :, :], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                next_state = self.model(history_tensor).detach().cpu().numpy()[0]
            predictions.append(next_state)
            current_history = np.concatenate([current_history[1:], next_state[None, :]], axis=0)
        return np.stack(predictions, axis=0)

    def compute_prediction_confidence(
        self,
        history: np.ndarray,
        horizon_seconds: int = 10,
        step_seconds: float = 1.0,
        mc_samples: int = 25,
    ) -> PredictionConfidence:
        self.model.train()
        sampled_predictions: list[np.ndarray] = []
        for _sample in range(mc_samples):
            sampled_predictions.append(self.predict_next_state(history, horizon_seconds=horizon_seconds, step_seconds=step_seconds))
        sampled = np.stack(sampled_predictions, axis=0)
        return PredictionConfidence(
            mean_prediction=sampled.mean(axis=0),
            lower_bound=np.quantile(sampled, 0.05, axis=0),
            upper_bound=np.quantile(sampled, 0.95, axis=0),
            std_prediction=sampled.std(axis=0),
        )

    def detect_anomaly(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted_arr = np.asarray(predicted, dtype=np.float32)
        actual_arr = np.asarray(actual, dtype=np.float32)
        rmse = float(np.sqrt(np.mean((predicted_arr - actual_arr) ** 2)))
        score = 1.0 - math.exp(-rmse / 0.12)
        return float(np.clip(score, 0.0, 1.0))


def train_dt_predictor(
    *,
    output_dir: Path,
    device: str = "cpu",
    num_sequences: int = 256,
    sequence_length: int = 20,
    input_dim: int = 384,
    prediction_horizon: int = 10,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences, future_states = simulate_dt_sequences(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        input_dim=input_dim,
        prediction_horizon=prediction_horizon,
        random_seed=random_seed,
    )
    train_cutoff = int(num_sequences * 0.8)
    predictor = DTPredictor(input_dim=input_dim, device=device)
    fit_summary = predictor.fit(
        sequences[:train_cutoff],
        future_states[:train_cutoff, 0, :],
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    mse_rows: list[dict[str, float]] = []
    for sequence, actual_future in zip(sequences[train_cutoff:], future_states[train_cutoff:]):
        predicted_future = predictor.predict_next_state(sequence, horizon_seconds=prediction_horizon, step_seconds=1.0)
        mse_rows.append(
            {
                "prediction_mse": float(np.mean((predicted_future - actual_future) ** 2)),
            }
        )
    mse_df = pd.DataFrame(mse_rows)
    write_dataframe(output_dir / "dt_prediction_mse.csv", mse_df)

    checkpoint_path = output_dir / "dt_predictor.pt"
    torch.save({"state_dict": predictor.model.state_dict(), "input_dim": input_dim}, checkpoint_path)
    summary_payload = {
        "fit_summary": fit_summary,
        "mean_prediction_mse": float(mse_df["prediction_mse"].mean()),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }
    write_json(output_dir / "dt_predictor_summary.json", summary_payload)
    return {
        "predictor": predictor,
        "mean_prediction_mse": float(mse_df["prediction_mse"].mean()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "summary_json": str((output_dir / "dt_predictor_summary.json").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the predictive digital-twin LSTM on simulation-derived histories.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-subdir", default="dt_predictor_training")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-sequences", type=int, default=384)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--prediction-horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    output_dir = paths.outputs_csv_paper1 / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = train_dt_predictor(
        output_dir=output_dir,
        device=args.device,
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        input_dim=384,
        prediction_horizon=args.prediction_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    print(f"Predictive DT checkpoint: {results['checkpoint_path']}")
    print(f"Mean prediction MSE: {results['mean_prediction_mse']:.4f}")
    print(f"Summary JSON: {results['summary_json']}")


if __name__ == "__main__":
    main()
