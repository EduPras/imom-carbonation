import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from pathlib import Path
from .base_model import BaseModel
import numpy as np


class ConcretePredictor(nn.Module):
    def __init__(self, input_dim: int = 9):
        super(ConcretePredictor, self).__init__()

        # Input layer: input_dim features -> 32 neurons
        self.fc1 = nn.Linear(input_dim, 32)
        # Hidden layer: 32 neurons -> 16 neurons
        self.fc2 = nn.Linear(32, 16)
        # Output layer: 16 neurons -> 1 target
        self.fc3 = nn.Linear(16, 1)

        # Activation and Regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MLPModel(BaseModel):
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 1e-04,
        patience: int = 50,
        checkpoint_dir: str = "checkpoints",
        epochs: int = 500,
        seed: int = 42,
        input_dim: int = 9,
    ) -> None:
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.info(f"Initializing MLP on device: {self.device}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.seed = seed
        self.input_dim = input_dim

        self.model = ConcretePredictor(input_dim=self.input_dim).to(self.device)
        self.criterion = nn.MSELoss()

        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15
        )

    def train(
        self,
        X_train: torch.FloatTensor,
        y_train: torch.FloatTensor,
        X_test: torch.FloatTensor,
        y_test: torch.FloatTensor,
    ) -> None:
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting MLP training loop...")
        
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.model.train()

            # Forward pass
            predictions = self.model(X_train)
            loss = self.criterion(predictions, y_train)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_test)
                val_loss = self.criterion(val_preds, y_test)
                
            history["train_loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss.item()))

            self.scheduler.step(val_loss)

            # Checkpointing & Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss.item(),
                }
                torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")

            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                logger.debug(
                    f"Epoch [{epoch + 1}/{self.epochs}] | Train Loss: {
                        loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Patience: {
                        patience_counter
                    }/{self.patience}"
                )

            if patience_counter >= self.patience:
                logger.warning(
                    f"Early stopping triggered at Epoch {
                        epoch + 1
                    }. No improvement for {self.patience} epochs."
                )
                break

        logger.success(
            f"MLP Training complete. Best Validation Loss: {best_val_loss:.4f}"
        )

        # Load best model for future predictions
        self.model.load_state_dict(
            torch.load(self.checkpoint_dir / "best_model.pth", weights_only=True)[
                "model_state_dict"
            ]
        )
        self.model.eval()
        return history

    def predict(self, X: torch.FloatTensor) -> np.ndarray:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            preds = self.model(X)
        return preds.cpu().numpy()

    def get_name(self) -> str:
        return "MLP"

    def reset(self) -> None:
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        self.model = ConcretePredictor(input_dim=self.input_dim).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15
        )

    def save(self, path: Path) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
        }, path.with_suffix('.pth'))


