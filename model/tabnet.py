import torch
import numpy as np
import optuna
from pathlib import Path
from loguru import logger
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel



class TabNetModel(BaseModel):
    def __init__(self, seed: int = 42, n_trials: int = 15):
        self.seed = seed
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train, X_test, y_test):
        n_da = trial.suggest_int("n_da", 8, 32)
        n_steps = trial.suggest_int("n_steps", 3, 7)
        gamma = trial.suggest_float("gamma", 1.0, 1.8)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-4, 1e-1, log=True)

        model = TabNetRegressor(
            n_d=n_da,
            n_a=n_da,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            seed=self.seed,
            verbose=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["mse"],
            max_epochs=100,
            patience=15,
        )
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

        logger.info("Starting Optuna tuning for TabNet...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(
                trial, X_train_np, y_train_np, X_test_np, y_test_np
            ),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params
        logger.info(f"Best TabNet Params: {self.best_params}")

        n_da = self.best_params["n_da"]
        self.model = TabNetRegressor(
            n_d=n_da,
            n_a=n_da,
            n_steps=self.best_params["n_steps"],
            gamma=self.best_params["gamma"],
            lambda_sparse=self.best_params["lambda_sparse"],
            seed=self.seed,
            verbose=0,
        )
        self.model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_train_np, y_train_np), (X_test_np, y_test_np)],
            eval_name=["train", "val"],
            eval_metric=["mse"],
            max_epochs=150,
            patience=25,
        )
        return {
            "train_loss": self.model.history["train_mse"],
            "val_loss": self.model.history["val_mse"]
        }

    def predict(self, X: torch.FloatTensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X.numpy())

    def get_name(self) -> str:
        return "TabNet"

    def reset(self) -> None:
        self.model = None

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save_model(str(path))

