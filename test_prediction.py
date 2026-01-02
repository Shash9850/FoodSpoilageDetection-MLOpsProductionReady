from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.config import Config
import os

config = Config()

model_path = "artifacts/models/food_spoilage_cnn.pth"
raw_data_dir = config.get_artifact_dir()["raw_data_dir"]

# IMPORTANT: get class names from raw data folders
class_names = sorted(os.listdir(raw_data_dir))

predictor = PredictionPipeline(model_path, class_names)

# CHANGE THIS to any one image path
test_image_path = os.path.join(raw_data_dir, class_names[0], os.listdir(os.path.join(raw_data_dir, class_names[0]))[0])

prediction = predictor.predict(test_image_path)
print("Predicted class:", prediction)
