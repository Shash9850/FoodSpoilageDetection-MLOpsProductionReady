import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger = get_logger()

class FoodSpoilageCNN(nn.Module):
    def __init__(self, num_classes):
        super(FoodSpoilageCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelTrainer:
    def __init__(self, config, train_loader):
        try:
            self.train_loader = train_loader
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.num_classes = len(train_loader.dataset.class_names)

            self.model = FoodSpoilageCNN(self.num_classes).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.get_training_params()["learning_rate"]
            )
            self.epochs = config.get_training_params()["num_epochs"]

            self.model_dir = config.get_artifact_dir()["model_dir"]
            os.makedirs(self.model_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e, sys)

    def train(self):
        try:
            logger.info("Starting model training...")

            for epoch in range(self.epochs):
                running_loss = 0.0

                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(self.train_loader)
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            model_path = os.path.join(self.model_dir, "food_spoilage_cnn.pth")
            torch.save(self.model.state_dict(), model_path)

            logger.info(f"Model training completed. Saved at {model_path}")

        except Exception as e:
            raise CustomException(e, sys)
