import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from src.components.model_trainer import FoodSpoilageCNN
from src.utils.exception import CustomException
from src.utils.logger import get_logger
import sys

logger = get_logger()

class PredictionPipeline:
    def __init__(self, model_path, class_names):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.class_names = class_names

            self.model = FoodSpoilageCNN(num_classes=len(class_names))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            logger.info("Prediction pipeline initialized successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs, 1)

            prediction = self.class_names[predicted.item()]
            logger.info(f"Prediction completed: {prediction}")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
