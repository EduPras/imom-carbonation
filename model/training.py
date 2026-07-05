from pathlib import Path
from loguru import logger
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .loader import DataLoader
from .base_model import BaseModel


def log_metrics(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(
        f"[{model_name}] Metrics - MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
    )
    return mse, mae, r2


class TrainingOrchestrator:
    def __init__(self, filepath: Path, seed: int = 42):
        self.filepath = filepath
        self.seed = seed
        self.data_loader = DataLoader(csv_path=filepath, seed=seed)

        # Load and preprocess data once
        self.X_train, self.y_train, self.X_test, self.y_test = (
            self.data_loader.preprocess_data()
        )

    def train_and_evaluate(self, model: BaseModel):
        model_name = model.get_name()
        logger.info(f"--- Training {model_name} ---")

        # Train model
        model.train(self.X_train, self.y_train, self.X_test, self.y_test)

        # Evaluate model
        preds = model.predict(self.X_test)

        # Convert true labels to numpy if they are tensors
        y_test_np = (
            self.y_test.cpu().numpy()
            if isinstance(self.y_test, torch.Tensor)
            else self.y_test
        )

        log_metrics(model_name, y_test_np, preds)
