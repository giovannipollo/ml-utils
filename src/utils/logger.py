import os
import logging

def get_logger(log_dir, log_name):
    """
    Create and configure a logger with file and console handlers.

    Args:
        log_dir (str): Directory where the log file will be created.
        log_name (str): Name of the logger and the log file.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger