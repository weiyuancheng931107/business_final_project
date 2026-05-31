"""
utils.py
Shared utilities for the Financial Sentiment Backtesting System.
"""
from __future__ import annotations

import functools
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional, Tuple, TypeVar

# ──────────────────────────────────────────────────────────────
# Terminal colour helpers
# ──────────────────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    _COLORAMA = True
except ImportError:
    _COLORAMA = False

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_RED     = "\033[91m"
_GREEN   = "\033[92m"
_YELLOW  = "\033[93m"
_CYAN    = "\033[96m"
_BLUE    = "\033[94m"
_MAGENTA = "\033[95m"
_WHITE   = "\033[97m"


def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty() and not _COLORAMA:
        return text
    return f"{code}{text}{_RESET}"


def ok(text: str)     -> str: return _c(text, _GREEN)
def warn(text: str)   -> str: return _c(text, _YELLOW)
def err(text: str)    -> str: return _c(text, _RED)
def info(text: str)   -> str: return _c(text, _CYAN)
def bold(text: str)   -> str: return _c(text, _BOLD)
def header(text: str) -> str: return _c(text, _BLUE + _BOLD)
def accent(text: str) -> str: return _c(text, _MAGENTA)


def print_section(title: str) -> None:
    bar = "─" * 60
    print(f"\n{header(bar)}")
    print(f"{header('  ' + title)}")
    print(f"{header(bar)}")


def print_metric(label: str, value: Any, unit: str = "") -> None:
    label_s = f"{info(label + ':'):<30}"
    value_s = bold(str(value))
    print(f"  {label_s} {value_s} {unit}")


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_LOG_DIR, exist_ok=True)


def get_logger(name: str = "backtest") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(_LOG_DIR, f"backtest_{ts}.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────────
# Retry decorator
# ──────────────────────────────────────────────────────────────
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.5,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exc: Exception = RuntimeError("Unknown error")
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        msg = (
                            f"[retry {attempt}/{max_attempts}] "
                            f"{func.__name__} failed: {exc}. "
                            f"Retrying in {_delay:.1f}s…"
                        )
                        if logger:
                            logger.warning(msg)
                        else:
                            print(warn(msg))
                        time.sleep(_delay)
                        _delay *= backoff
            raise last_exc
        return wrapper  # type: ignore[return-value]
    return decorator


# ──────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────

def wilson_ci(
    n_success: int,
    n_total: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """
    Wilson score 95% confidence interval for a proportion.

    Returns (lower, upper) as fractions in [0, 1].
    More accurate than the normal approximation, especially for
    small samples or proportions near 0 or 1.

    Parameters
    ----------
    n_success : number of correct predictions
    n_total   : total predictions
    z         : z-score for desired confidence level (1.96 = 95%)
    """
    if n_total == 0:
        return (0.0, 0.0)
    p = n_success / n_total
    z2 = z * z
    n = n_total
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def format_ci(lower: float, upper: float, pct: bool = True) -> str:
    """Format a confidence interval for display."""
    if pct:
        return f"[{lower*100:.1f}%, {upper*100:.1f}%]"
    return f"[{lower:.4f}, {upper:.4f}]"


# ──────────────────────────────────────────────────────────────
# Direction / sentiment mappings
# ──────────────────────────────────────────────────────────────

SENTIMENT_TO_DIRECTION: dict[str, str] = {
    "positive":          "bullish",
    "slightly_positive": "weak_bullish",
    "neutral":           "neutral",
    "slightly_negative": "weak_bearish",
    "negative":          "bearish",
}

BULLISH_THRESHOLD      =  2.0
WEAK_BULLISH_THRESHOLD =  0.5
WEAK_BEARISH_THRESHOLD = -0.5
BEARISH_THRESHOLD      = -2.0

# All prediction horizons (trading days)
FUTURE_DAYS = [1, 3, 5, 10, 20, 60]


def return_to_direction(pct_return: float) -> str:
    if pct_return >= BULLISH_THRESHOLD:
        return "bullish"
    if pct_return >= WEAK_BULLISH_THRESHOLD:
        return "weak_bullish"
    if pct_return >= WEAK_BEARISH_THRESHOLD:
        return "neutral"
    if pct_return >= BEARISH_THRESHOLD:
        return "weak_bearish"
    return "bearish"


_BULLISH_DIRECTIONS = {"bullish", "weak_bullish"}
_BEARISH_DIRECTIONS = {"bearish", "weak_bearish"}


def predicted_is_bullish(direction: str) -> Optional[bool]:
    if direction in _BULLISH_DIRECTIONS:
        return True
    if direction in _BEARISH_DIRECTIONS:
        return False
    return None


def binary_correct(predicted_direction: str, actual_direction: str) -> Optional[bool]:
    """Return True/False for directional correctness, None if one is neutral."""
    pred = predicted_is_bullish(predicted_direction)
    act  = predicted_is_bullish(actual_direction)
    if pred is None or act is None:
        return None
    return pred == act


def sentiment_label_pretty(label: str) -> str:
    mapping = {
        "positive":          ok("正向 ↑↑"),
        "slightly_positive": ok("偏正 ↑"),
        "neutral":           info("中性 →"),
        "slightly_negative": warn("偏負 ↓"),
        "negative":          err("負向 ↓↓"),
    }
    return mapping.get(label, label)


def direction_pretty(direction: str) -> str:
    mapping = {
        "bullish":      ok("bullish ↑↑"),
        "weak_bullish": ok("weak_bullish ↑"),
        "neutral":      info("neutral →"),
        "weak_bearish": warn("weak_bearish ↓"),
        "bearish":      err("bearish ↓↓"),
    }
    return mapping.get(direction, direction)


# ──────────────────────────────────────────────────────────────
# File helpers
# ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def results_path(filename: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


def timestamped_name(base: str, ext: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}.{ext}"