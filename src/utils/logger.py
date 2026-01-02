import os
from loguru import logger
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)

logger.add(
    LOG_FILE,
    format="{time} | {level} | {message}",
    level="INFO",
    rotation="1 MB"
)

def get_logger():
    return logger
