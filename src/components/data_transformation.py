import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger = get_logger()

class FoodDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        try:
            self.image_dir = image_dir
            self.transform = transform
            self.images = []
            self.labels = []

            # Map class names to numeric labels
            self.class_names = sorted(os.listdir(image_dir))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

            for cls_name in self.class_names:
                cls_folder = os.path.join(image_dir, cls_name)
                for img_name in os.listdir(cls_folder):
                    self.images.append(os.path.join(cls_folder, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
            logger.info(f"Found {len(self.images)} images across {len(self.class_names)} classes.")

        except Exception as e:
            raise CustomException(e, sys)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path = self.images[idx]
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]
            return image, label

        except Exception as e:
            raise CustomException(e, sys)


def get_data_loader(image_dir, batch_size=16, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = FoodDataset(image_dir=image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
