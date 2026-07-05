from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Union


class BaseModel(ABC):
    @abstractmethod
    def train(
        self,
        X_train: torch.FloatTensor,
        y_train: torch.FloatTensor,
        X_test: torch.FloatTensor,
        y_test: torch.FloatTensor,
    ) -> None:
        """Trains the model using the provided data."""
        pass

    @abstractmethod
    def predict(self, X: torch.FloatTensor) -> Union[torch.FloatTensor, np.ndarray]:
        """Generates predictions for the given data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns the model name."""
        pass
