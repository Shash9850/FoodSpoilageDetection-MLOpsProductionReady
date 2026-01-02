from src.components.data_ingestion import DataIngestion
from src.utils.config import Config

config = Config()
data_ingestion = DataIngestion(config)


source_dataset_dir = "E:\Placement\Projects\FoodSpoilageProductionReadyProject\data\Train"

data_ingestion.ingest_data(source_dataset_dir)
