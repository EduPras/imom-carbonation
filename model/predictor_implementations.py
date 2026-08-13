import torch
import numpy as np
import joblib
from pathlib import Path

from .base_predictor import BasePredictor
from .mlp import ConcretePredictor
from pytorch_tabnet.tab_model import TabNetRegressor


class GeneralPredictor(BasePredictor):
    """A generic predictor for models that can be serialized/deserialized with joblib."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.folds = []

    def load(self, checkpoints_dir: Path) -> None:
        self.folds = []
        model_dir = checkpoints_dir / self.model_name
        scalers_dir = checkpoints_dir / "scalers"

        # Determine K by scanning the directory
        fold_files = list(model_dir.glob("fold_*.joblib"))
        k = len(fold_files)

        for i in range(1, k + 1):
            model_path = model_dir / f"fold_{i}.joblib"
            scaler_x_path = scalers_dir / f"fold_{i}_X.pkl"
            scaler_y_path = scalers_dir / f"fold_{i}_y.pkl"

            model = joblib.load(model_path)
            scaler_x = joblib.load(scaler_x_path)
            scaler_y = joblib.load(scaler_y_path)

            self.folds.append((model, scaler_x, scaler_y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for model, scaler_x, scaler_y in self.folds:
            X_scaled = scaler_x.transform(X)
            y_pred_scaled = model.predict(X_scaled)
            if y_pred_scaled.ndim == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            preds.append(y_pred)

        return np.mean(preds, axis=0)

    def get_name(self) -> str:
        return self.model_name


class MLPPredictor(BasePredictor):
    def __init__(self):
        self.folds = []
        self.device = torch.device("cpu")

    def load(self, checkpoints_dir: Path) -> None:
        self.folds = []
        model_dir = checkpoints_dir / "MLP"
        scalers_dir = checkpoints_dir / "scalers"

        fold_files = list(model_dir.glob("fold_*.pth"))
        k = len(fold_files)

        for i in range(1, k + 1):
            model_path = model_dir / f"fold_{i}.pth"
            scaler_x_path = scalers_dir / f"fold_{i}_X.pkl"
            scaler_y_path = scalers_dir / f"fold_{i}_y.pkl"

            scaler_x = joblib.load(scaler_x_path)
            scaler_y = joblib.load(scaler_y_path)

            input_dim = scaler_x.n_features_in_
            model = ConcretePredictor(input_dim=input_dim).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.folds.append((model, scaler_x, scaler_y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for model, scaler_x, scaler_y in self.folds:
            X_scaled = scaler_x.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                y_pred_scaled_tensor = model(X_tensor)
            y_pred_scaled = y_pred_scaled_tensor.cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            preds.append(y_pred)

        return np.mean(preds, axis=0)

    def get_name(self) -> str:
        return "MLP"


class TabNetPredictor(BasePredictor):
    def __init__(self):
        self.folds = []

    def load(self, checkpoints_dir: Path) -> None:
        self.folds = []
        model_dir = checkpoints_dir / "TabNet"
        scalers_dir = checkpoints_dir / "scalers"

        # TabNet saves zip files
        fold_files = list(model_dir.glob("fold_*.zip"))
        k = len(fold_files)

        for i in range(1, k + 1):
            model_path = model_dir / f"fold_{i}.zip"
            scaler_x_path = scalers_dir / f"fold_{i}_X.pkl"
            scaler_y_path = scalers_dir / f"fold_{i}_y.pkl"


            model = TabNetRegressor()
            model.load_model(str(model_path))

            scaler_x = joblib.load(scaler_x_path)
            scaler_y = joblib.load(scaler_y_path)

            self.folds.append((model, scaler_x, scaler_y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for model, scaler_x, scaler_y in self.folds:
            X_scaled = scaler_x.transform(X)
            y_pred_scaled = model.predict(X_scaled)
            if y_pred_scaled.ndim == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            preds.append(y_pred)

        return np.mean(preds, axis=0)

    def get_name(self) -> str:
        return "TabNet"


def get_all_predictors(checkpoints_dir: Path = Path("checkpoints")) -> list[BasePredictor]:
    predictors = [
        MLPPredictor(),
        GeneralPredictor("Random Forest"),
        GeneralPredictor("XGBoost"),
        GeneralPredictor("LightGBM"),
        GeneralPredictor("CatBoost"),
        GeneralPredictor("SVR"),
        GeneralPredictor("Ridge Regression"),
        TabNetPredictor(),
    ]

    for predictor in predictors:
        predictor.load(checkpoints_dir)

    return predictors
