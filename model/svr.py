import torch
import numpy as np
import optuna
import joblib
from pathlib import Path
from loguru import logger
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from .base_model import BaseModel



class SVRModel(BaseModel):
    def __init__(self, seed: int = 42, n_trials: int = 20):
        self.seed = seed
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train, X_test, y_test):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-4, 1.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)

        base_model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel="rbf")
        model = MultiOutputRegressor(base_model)
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

        logger.info("Starting Optuna tuning for SVR...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(
                trial, X_train_np, y_train_np, X_test_np, y_test_np
            ),
            n_trials=self.n_trials,
        )

        self.best_params = study.best_params
        logger.info(f"Best SVR Params: {self.best_params}")

        base_model = SVR(**self.best_params, kernel="rbf")
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train_np, y_train_np)
        
        train_preds = self.model.predict(X_train_np)
        test_preds = self.model.predict(X_test_np)
        from sklearn.metrics import mean_squared_error
        return {
            "train_loss": [float(mean_squared_error(y_train_np, train_preds))],
            "val_loss": [float(mean_squared_error(y_test_np, test_preds))]
        }

    def predict(self, X: torch.FloatTensor) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X.numpy())

    def get_name(self) -> str:
        return "SVR"

    def reset(self) -> None:
        self.model = None

    def save(self, path: Path) -> None:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        joblib.dump(self.model, path.with_suffix('.joblib'))

