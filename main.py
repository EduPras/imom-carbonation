from pathlib import Path
from loguru import logger

from model.training import TrainingOrchestrator
from model.mlp import MLPModel
from model.rf import RFModel
from model.xgb import XGBModel

if __name__ == "__main__":
    path = Path("cube_strength_carbonation_depth_data.csv")

    orchestrator = TrainingOrchestrator(path)

    models = [MLPModel(), RFModel(), XGBModel()]

    for model in models:
        orchestrator.train_and_evaluate(model)
