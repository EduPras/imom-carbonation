import torch
import numpy as np
import optuna
import joblib
from pathlib import Path
from loguru import logger
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel



class XGBModel(BaseModel):
    def __init__(self, seed: int = 42, n_trials: int = 20):
        self.seed = seed
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train, X_test, y_test):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.seed,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return mse

    def train(
        self,
        X_train: torch.FloatTensor,
        y_train: torch.FloatTensor,
        X_test: torch.FloatTensor,
        y_test: torch.FloatTensor,
    ) -> None:
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy()
        X_test_np = X_test.numpy()
        y_test_np = y_test.numpy()

        logger.info("Starting Optuna tuning for XGBoost...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(
                trial, X_train_np, y_train_np, X_test_np, y_test_np
            ),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params
        logger.info(f"Best XGB Params: {self.best_params}")

        self.model = XGBRegressor(**self.best_params, random_state=self.seed)
        self.model.fit(X_train_np, y_train_np, eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)], verbose=False)
        results = self.model.evals_result()
        try:
            return {
                "train_loss": list(results["validation_0"].values())[0],
                "val_loss": list(results["validation_1"].values())[0]
            }
        except Exception:
            return {"train_loss": [], "val_loss": []}

    def predict(self, X: torch.FloatTensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X.numpy())

    def get_name(self) -> str:
        return "XGBoost"

    def reset(self) -> None:
        self.model = None

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, path.with_suffix('.joblib'))


