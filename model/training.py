from pathlib import Path
from loguru import logger
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .loader import DataLoader
from .base_model import BaseModel


def to_numpy(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return np.asarray(data)


def log_metrics(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2


class TrainingOrchestrator:
    def __init__(self, filepath: Path, seed: int = 42, n_splits: int = 5):
        self.filepath = filepath
        self.seed = seed
        self.n_splits = n_splits
        self.data_loader = DataLoader(csv_path=filepath, seed=seed)

        # Retrieve raw X and y from DataLoader
        if self.data_loader.data is None:
            raise ValueError("DataLoader failed to load data.")

        self.X, self.y = self.data_loader.data
        self.X_np = self.X.values
        self.y_np = self.y.values

    def train_and_evaluate(self, model: BaseModel):
        model_name = model.get_name()
        logger.info(f"--- Training {model_name} ---")
        logger.info(
            f"Starting {self.n_splits}-Fold Cross-Validation for {model_name}..."
        )

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        mses = []
        maes = []
        r2s = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(self.X_np)):
            # Split data
            X_train_fold, X_test_fold = self.X_np[train_idx], self.X_np[test_idx]
            y_train_fold, y_test_fold = self.y_np[train_idx], self.y_np[test_idx]

            # Fit scalers on this fold's training data to avoid leakage
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train_fold)
            X_test_scaled = scaler_X.transform(X_test_fold)
            y_train_scaled = scaler_y.fit_transform(y_train_fold)
            y_test_scaled = scaler_y.transform(y_test_fold)

            # Convert to PyTorch FloatTensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)

            # Reset model state for this fold
            model.reset()

            # Train model
            model.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

            # Predict
            preds = model.predict(X_test_tensor)

            # Standardize predictions and ground truth type
            y_test_np = to_numpy(y_test_tensor)
            preds_np = to_numpy(preds)

            # Calculate metrics
            mse, mae, r2 = log_metrics(model_name, y_test_np, preds_np)
            logger.info(
                f"[{model_name}] Fold {fold + 1}/{self.n_splits} - MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
            )

            mses.append(mse)
            maes.append(mae)
            r2s.append(r2)

        # Compute summary stats
        mean_mse, std_mse = np.mean(mses), np.std(mses)
        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_r2, std_r2 = np.mean(r2s), np.std(r2s)

        logger.success(f"{model_name} {self.n_splits}-Fold CV Completed.")
        logger.info(
            f"\n[{model_name}] CV Metrics Summary:\n"
            f"  - MSE: {mean_mse:.4f} +/- {std_mse:.4f}\n"
            f"  - MAE: {mean_mae:.4f} +/- {std_mae:.4f}\n"
            f"  - R2:  {mean_r2:.4f} +/- {std_r2:.4f}"
        )

