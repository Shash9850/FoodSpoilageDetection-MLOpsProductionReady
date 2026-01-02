from src.components.data_transformation import get_data_loader
from src.components.model_trainer import ModelTrainer
from src.utils.config import Config

config = Config()
raw_data_dir = config.get_artifact_dir()["raw_data_dir"]

train_loader = get_data_loader(raw_data_dir, batch_size=16)
trainer = ModelTrainer(config, train_loader)

trainer.train()
