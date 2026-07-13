import torch
import numpy as np
import optuna
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel


class RidgeModel(BaseModel):
    def __init__(self, seed: int = 42, n_trials: int = 20):
        self.seed = seed
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train, X_test, y_test):
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)

        model = Ridge(alpha=alpha, random_state=self.seed)
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

        logger.info("Starting Optuna tuning for Ridge Regression...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(
                trial, X_train_np, y_train_np, X_test_np, y_test_np
            ),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params
        logger.info(f"Best Ridge Params: {self.best_params}")

        self.model = Ridge(**self.best_params, random_state=self.seed)
        self.model.fit(X_train_np, y_train_np)

    def predict(self, X: torch.FloatTensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X.numpy())

    def get_name(self) -> str:
        return "Ridge Regression"

    def reset(self) -> None:
        self.model = None
