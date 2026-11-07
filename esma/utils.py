import logging
import random
import re
import string
import sys

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Seed everything for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_metrics(metrics: dict[str, float], decimal_places: int = 4) -> str:
    """Format a dict of metric name -> value as a single comma-separated string."""
    return ", ".join([f"{k}: {v:.{decimal_places}f}" for k, v in metrics.items()])


def get_logger(name: str) -> logging.Logger:
    """Return logger for logging

    Args:
        name: logger name
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def replace_underscore(text: str) -> str:
    return text.replace("_", " ")


# Official TriviaQA also excludes curly quotes and the acute accent, and replaces each
# punctuation character with a space rather than deleting it (so "well-known" -> "well known").
_PUNC_EXCLUDE = set(string.punctuation + "‘’´`")


def handle_punc(text: str) -> str:
    return "".join(ch if ch not in _PUNC_EXCLUDE else " " for ch in text)


def normalize_answer(s: str) -> str:
    """Normalize an answer using the official TriviaQA exact-match convention."""
    return white_space_fix(remove_articles(handle_punc(replace_underscore(s.lower())))).strip()
