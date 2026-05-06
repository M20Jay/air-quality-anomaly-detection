import logging
import os
from datetime import datetime

def get_logger(name:str) -> logging.Logger:
    """
    Create and return a configured logger.
    
    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Data loaded successfully")
        logger.error("Failed to connect to database")
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    # Create logger
    logger=logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    # Format: timestamp | level | module | message
    formatter=logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — prints to terminal
    console_handler=logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler — writes to logs/app.log
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler=logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger