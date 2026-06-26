from pathlib import Path
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, csv_path: Path, seed: int = 42) -> None:
        self.path: Path = csv_path
        self.seed: int = seed

        self.data = self._load_csv()

    @staticmethod
    def _display_stats(tensor_name: str, t: torch.FloatTensor):
        logger.info(
            f"[{tensor_name}]: Shape: {t.shape}\nMeans: {t.mean(dim=0)}\nStd: {
                t.std(dim=0)}"
        )

    def preprocess_data(self) -> list[torch.FloatTensor]:
        if self.data is None:
            logger.error("Could not load the data")
            return []

        X, y = self.data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.seed
            )
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train)
            y_test_scaled = scaler_y.transform(y_test)

            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)

            self._display_stats("X_train", X_train_tensor)
            self._display_stats("y_train", y_train_tensor)
            self._display_stats("X_test", X_test_tensor)
            self._display_stats("y_test", X_train_tensor)

            return [X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor]
        except Exception as e:
            logger.exception(f"Failed to split data: {e}")
            return []

    def _load_csv(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        try:
            df = pd.read_csv(self.path)
            # drops the first column (index)
            df = df.iloc[:, 1:]
            logger.debug(f"CSV info: {df.shape} (data points, columns)")

            X = df.iloc[:, :-2]
            Y = df.iloc[:, -2:]

            logger.debug(f"X columns: {X.columns}")
            logger.debug(f"Y columns: {Y.columns}")
            return X, Y
        except Exception as e:
            logger.exception(f"Failed to read CSV file: {e}")
            return None

    def scale_data(self) -> None: ...


if __name__ == "__main__":
    loader = DataLoader(Path("cube_strength_carbonation_depth_data.csv"))
    data = loader.preprocess_data()
