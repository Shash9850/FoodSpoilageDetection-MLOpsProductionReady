import os
import shutil
from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger = get_logger()

class DataIngestion:
    def __init__(self, config):
        try:
            self.raw_data_dir = config.get_artifact_dir()["raw_data_dir"]
            self.processed_data_dir = config.get_artifact_dir()["processed_data_dir"]
            os.makedirs(self.raw_data_dir, exist_ok=True)
            os.makedirs(self.processed_data_dir, exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)

    def ingest_data(self, source_dir):
        """
        Copies images from source_dir (raw dataset) to artifacts/raw
        """
        try:
            if not os.path.exists(source_dir):
                raise Exception(f"Source directory {source_dir} not found")

            for folder_name in os.listdir(source_dir):
                folder_path = os.path.join(source_dir, folder_name)
                if os.path.isdir(folder_path):
                    dest_folder = os.path.join(self.raw_data_dir, folder_name)
                    os.makedirs(dest_folder, exist_ok=True)

                    for file in os.listdir(folder_path):
                        src_file = os.path.join(folder_path, file)
                        dst_file = os.path.join(dest_folder, file)
                        shutil.copy(src_file, dst_file)

            logger.info(f"Data ingestion completed. Raw data is at {self.raw_data_dir}")

        except Exception as e:
            raise CustomException(e, sys)
