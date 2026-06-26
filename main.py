from model.training import Trainer
from pathlib import Path

trainer = Trainer(Path("cube_strength_carbonation_depth_data.csv"))
trainer.train()
