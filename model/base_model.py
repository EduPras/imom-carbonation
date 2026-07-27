from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Union
from pathlib import Path



class BaseModel(ABC):
    @abstractmethod
    def train(
        self,
        X_train: torch.FloatTensor,
        y_train: torch.FloatTensor,
        X_test: torch.FloatTensor,
        y_test: torch.FloatTensor,
    ) -> dict:
        """Trains the model using the provided data and returns a history of metrics."""
        pass

    @abstractmethod
    def predict(self, X: torch.FloatTensor) -> Union[torch.FloatTensor, np.ndarray]:
        """Generates predictions for the given data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the model name."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the model state for a new training session (e.g. new fold)."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Saves the model state to the specified path."""
        pass


