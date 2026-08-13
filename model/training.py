from pathlib import Path
from loguru import logger
import torch
import numpy as np
import joblib
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
) -> tuple[float, float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2


class TrainingOrchestrator:
    def __init__(self, filepath: Path, seed: int = 42, n_splits: int = 5, checkpoints_dir: str = "checkpoints", drop_features: list[str] | None = None):
        self.filepath = filepath
        self.seed = seed
        self.n_splits = n_splits
        self.checkpoints_dir = Path(checkpoints_dir)
        self.data_loader = DataLoader(csv_path=filepath, seed=seed, drop_features=drop_features)

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
        rmses = []
        maes = []
        r2s = []

        is_tree_model = model_name in ["XGBoost", "LightGBM", "CatBoost", "Random Forest"]
        fold_shap = []
        fold_X_scaled = []

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

            # Train model and get history
            history = model.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

            # Save model and scalers for this fold
            scalers_dir = self.checkpoints_dir / "scalers"
            scalers_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler_X, scalers_dir / f"fold_{fold + 1}_X.pkl")
            joblib.dump(scaler_y, scalers_dir / f"fold_{fold + 1}_y.pkl")

            model_dir = self.checkpoints_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f"fold_{fold + 1}")
            
            if history:
                import json
                with open(model_dir / f"fold_{fold + 1}_history.json", "w") as f:
                    json.dump(history, f)

            # Predict
            preds = model.predict(X_test_tensor)

            # Standardize predictions and ground truth type
            y_test_np = to_numpy(y_test_tensor)
            preds_np = to_numpy(preds)

            # Calculate metrics
            mse, rmse, mae, r2 = log_metrics(model_name, y_test_np, preds_np)
            logger.info(
                f"[{model_name}] Fold {fold + 1}/{self.n_splits} - MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}"
            )

            mses.append(mse)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

            # Compute SHAP values for tree-based models on the validation split
            if is_tree_model:
                try:
                    import shap
                    raw_model = model.model
                    
                    if hasattr(raw_model, "estimators_"):
                        # Should not happen anymore, but just in case
                        estimator = raw_model.estimators_[0]
                        explainer = shap.TreeExplainer(estimator)
                        shap_val = explainer.shap_values(X_test_scaled)
                        if isinstance(shap_val, list):
                            shap_val = shap_val[0]
                        shap_val = np.expand_dims(shap_val, axis=-1)
                    else:
                        # Direct single-output tree models
                        explainer = shap.TreeExplainer(raw_model)
                        shap_val = explainer.shap_values(X_test_scaled)
                        
                        # Normalize shapes: ensure it is 3D (num_samples, num_features, 1)
                        if isinstance(shap_val, list):
                            shap_val = shap_val[0]
                        if len(shap_val.shape) == 2:
                            shap_val = np.expand_dims(shap_val, axis=-1)
                    
                    fold_shap.append(shap_val)
                    fold_X_scaled.append(X_test_scaled)
                except Exception as e:
                    logger.warning(f"[{model_name}] Failed to calculate SHAP for fold {fold + 1}: {e}")

        # Compute summary stats
        mean_mse, std_mse = np.mean(mses), np.std(mses)
        mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_r2, std_r2 = np.mean(r2s), np.std(r2s)

        logger.success(f"{model_name} {self.n_splits}-Fold CV Completed.")
        logger.info(
            f"\n[{model_name}] CV Metrics Summary:\n"
            f"  - MSE:  {mean_mse:.4f} +/- {std_mse:.4f}\n"
            f"  - RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}\n"
            f"  - MAE:  {mean_mae:.4f} +/- {std_mae:.4f}\n"
            f"  - R2:   {mean_r2:.4f} +/- {std_r2:.4f}"
        )

        # Aggregate and save out-of-fold SHAP values
        if is_tree_model and len(fold_shap) > 0:
            try:
                all_shap = np.concatenate(fold_shap, axis=0)
                all_X = np.concatenate(fold_X_scaled, axis=0)
                
                shap_data = {
                    "shap_values": all_shap,
                    "X_scaled": all_X,
                    "feature_names": list(self.X.columns),
                    "output_names": list(self.y.columns)
                }
                joblib.dump(shap_data, model_dir / "shap_data.pkl")
                logger.info(f"[{model_name}] Saved out-of-fold SHAP values to {model_dir / 'shap_data.pkl'}")
            except Exception as e:
                logger.error(f"[{model_name}] Failed to aggregate and save SHAP data: {e}")


