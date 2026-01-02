import yaml
from src.utils.exception import CustomException
import sys
import os

class Config:
    def __init__(self, config_file="config/config.yaml"):
        try:
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def get_artifact_dir(self):
        return self.config["artifacts"]

    def get_training_params(self):
        return self.config["training"]

    def get_model_params(self):
        return self.config["model"]

    def get_dataset_params(self):
        return self.config["dataset"]
