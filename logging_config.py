import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
APP_LOG_FILE = LOGS_DIR / "app.log"
ERROR_LOG_FILE = LOGS_DIR / "error.log"
ACCESS_LOG_FILE = LOGS_DIR / "access.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
JSON_LOG_FORMAT = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'

def setup_logging():
    """Configure logging with rotation and monitoring capabilities."""
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    json_formatter = logging.Formatter(JSON_LOG_FORMAT)

    # Create handlers
    # Main application log handler with rotation
    app_handler = logging.handlers.RotatingFileHandler(
        APP_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.INFO)

    # Error log handler with rotation
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    # Access log handler with rotation
    access_handler = logging.handlers.RotatingFileHandler(
        ACCESS_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    access_handler.setFormatter(json_formatter)
    access_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    # Configure access logger
    access_logger = logging.getLogger('access')
    access_logger.setLevel(logging.INFO)
    access_logger.addHandler(access_handler)

    # Configure application logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(app_handler)
    app_logger.addHandler(error_handler)
    app_logger.addHandler(console_handler)

    return app_logger, access_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)

# Initialize logging when module is imported
app_logger, access_logger = setup_logging() 