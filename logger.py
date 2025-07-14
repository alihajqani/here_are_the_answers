import sys
import logging
from logging.handlers import RotatingFileHandler

_default_log_file = "app.log"


def get_logger(
    name: str,
    log_file: str = _default_log_file,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Returns a logger configured to output to both console and a rotating file.

    Args:
        name (str): Logger name, typically __name__ of the module.
        log_file (str): Path to the log file.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.
        max_bytes (int): Maximum size in bytes before rotating the log file.
        backup_count (int): Number of backup files to keep.

    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.error("Error message")
    """
    logger = logging.getLogger(name)
    # Avoid adding handlers multiple times in interactive environments
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    fh.setLevel(file_level)
    fh.setFormatter(ch_formatter)
    logger.addHandler(fh)

    return logger
