from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BasePredictor(ABC):
    @abstractmethod
    def load(self, checkpoints_dir: Path) -> None:
        """Loads all K fold models and their corresponding scalers."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts output using ensembling of K fold models."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the model predictor."""
        pass
