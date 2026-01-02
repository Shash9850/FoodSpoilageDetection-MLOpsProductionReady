from src.utils.config import Config

config = Config()

print("Artifacts:", config.get_artifact_dir())
print("Training:", config.get_training_params())
print("Model:", config.get_model_params())
print("Dataset:", config.get_dataset_params())
