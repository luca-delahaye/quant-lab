"""Turn validated prices into returns.

Simple and log returns, and the two correct ways to accumulate them. Rule 4.2: the
first date is dropped, because N prices give N - 1 returns.
"""

import numpy as np
import pandas as pd

from quantlab.data import PriceDataError


class ReturnsError(ValueError):
    """Returns that cannot be trusted: an empty series, or a missing value inside it."""


def _check_prices(prices: pd.Series, positive_only: bool = False) -> None:
    """Refuse prices no return can be computed from."""
    if len(prices) < 2:
        raise PriceDataError(f"at least 2 prices are needed, got {len(prices)}")
    if prices.isna().any():
        raise PriceDataError(f"missing price on {prices.index[prices.isna()][0].date()}")
    bad = prices <= 0 if positive_only else prices == 0
    if bad.any():
        limit = "strictly positive" if positive_only else "non-zero"
        raise PriceDataError(f"prices must be {limit}: {prices.index[bad][0].date()}")


def _check_returns(returns: pd.Series) -> None:
    """Refuse returns nothing can be accumulated from."""
    if returns.empty:
        raise ReturnsError("no returns")
    if returns.isna().any():
        raise ReturnsError(f"missing return on {returns.index[returns.isna()][0].date()}")


def simple_returns(prices: pd.Series) -> pd.Series:
    """Simple returns, (today - yesterday) / yesterday, with the first date dropped."""
    _check_prices(prices)
    return (prices / prices.shift(1) - 1).iloc[1:]


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns, ln(today / yesterday), with the first date dropped.

    Needs strictly positive prices: ln(0) is -inf and ln of a negative is undefined.
    """
    _check_prices(prices, positive_only=True)
    return np.log(prices).diff().iloc[1:]


def cumulative_from_simple(simple: pd.Series) -> pd.Series:
    """Running cumulative return from simple returns: prod(1 + r) - 1.

    Never sum simple returns: on 100 -> 110 -> 100 that gives +0.91% instead of 0.
    """
    _check_returns(simple)
    return (1 + simple).cumprod() - 1


def cumulative_from_log(log: pd.Series) -> pd.Series:
    """Running cumulative return from log returns: exp(sum(l)) - 1.

    Log returns add up over time, which is what makes this correct.
    """
    _check_returns(log)
    return np.exp(log.cumsum()) - 1
