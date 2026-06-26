import torch.nn as nn
import torch
import torch.optim as optim
from pathlib import Path
from loguru import logger

from .model import ConcretePredictor
from .loader import DataLoader


class Trainer:
    def __init__(
        self,
        filepath: Path,
        lr: float = 0.01,
        weight_decay: float = 1e-04,
        patience: int = 50,
        checkpoint_dir: str = "checkpoints",
    ) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        logger.info(f"Initializing training on device: {self.device}")

        self.model = ConcretePredictor().to(self.device)
        self.criterion = nn.MSELoss()

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience

        data_loader = DataLoader(csv_path=filepath)
        X_train, y_train, X_test, y_test = data_loader.preprocess_data()

        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.X_test = X_test.to(self.device)
        self.y_test = y_test.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15
        )

    def train(self, epochs: int = 500):
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Starting training loop...")

        for epoch in range(epochs):
            self.model.train()

            # Forward pass
            predictions = self.model(self.X_train)
            loss = self.criterion(predictions, self.y_train)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Evaluate (Done every epoch for early stopping/scheduling)
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(self.X_test)
                val_loss = self.criterion(val_preds, self.y_test)

            self.scheduler.step(val_loss)

            # Checkpointing & Early Stopping Logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience

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
                    f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {
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

        logger.success(f"Training complete. Best Validation Loss: {
                       best_val_loss:.4f}")


if __name__ == "__main__":
    trainer = Trainer(Path("cube_strength_carbonation_depth_data.csv"))
    trainer.train()
