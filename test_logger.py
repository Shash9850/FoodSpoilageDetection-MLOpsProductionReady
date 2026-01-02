from src.utils.logger import get_logger
from src.utils.exception import CustomException
import sys

logger = get_logger()

try:
    logger.info("Testing logger...")
    x = 1 / 0
except Exception as e:
    logger.error("Exception occurred")
    raise CustomException(e, sys)
