import logging
from utils.config import load_config

def get_logger(name: str) -> logging.Logger:
    cfg = load_config().get("logging", {})
    level = getattr(logging, cfg.get("level", "INFO"))
    fmt = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
