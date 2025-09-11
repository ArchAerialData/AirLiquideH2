from __future__ import annotations
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter('[%(levelname)s] %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
