from src.components.data_transformation import get_data_loader
from src.utils.config import Config

config = Config()
raw_data_dir = config.get_artifact_dir()["raw_data_dir"]

loader = get_data_loader(raw_data_dir, batch_size=4)

for images, labels in loader:
    print(f"Batch images shape: {images.shape}, labels: {labels}")
    break  # Just test first batch
