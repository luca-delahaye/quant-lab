"""Turn validated prices into returns.

Simple and log returns, and the two correct ways to accumulate them. Rule 4.2: the
first date is dropped, because N prices give N - 1 returns.
"""

import numpy as np
import pandas as pd

from quantlab.data import PriceDataError


class ReturnsError(ValueError):
    """Returns that cannot be trusted: an empty series, or a missing value inside it."""


def check_prices(
    prices: pd.Series, min_length: int = 2, positive_only: bool = False
) -> None:
    """Refuse prices nothing can be computed from: min_length is 2 for returns, 1 for drawdown."""
    if len(prices) < min_length:
        raise PriceDataError(
            f"at least {min_length} price(s) are needed, got {len(prices)}"
        )
    if prices.isna().any():
        raise PriceDataError(f"missing price on {prices.index[prices.isna()][0].date()}")
    bad = prices <= 0 if positive_only else prices == 0
    if bad.any():
        limit = "strictly positive" if positive_only else "non-zero"
        raise PriceDataError(f"prices must be {limit}: {prices.index[bad][0].date()}")


def check_returns(returns: pd.Series) -> None:
    """Refuse returns nothing can be accumulated from."""
    if returns.empty:
        raise ReturnsError("no returns")
    if returns.isna().any():
        raise ReturnsError(f"missing return on {returns.index[returns.isna()][0].date()}")


def simple_returns(prices: pd.Series) -> pd.Series:
    """Simple returns, (today - yesterday) / yesterday, with the first date dropped."""
    check_prices(prices)
    return (prices / prices.shift(1) - 1).iloc[1:]


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns, ln(today / yesterday), with the first date dropped."""
    check_prices(prices, positive_only=True)
    return np.log(prices).diff().iloc[1:]


def cumulative_from_simple(simple: pd.Series) -> pd.Series:
    """Running cumulative return from simple returns: prod(1 + r) - 1."""
    check_returns(simple)
    return (1 + simple).cumprod() - 1


def cumulative_from_log(log: pd.Series) -> pd.Series:
    """Running cumulative return from log returns: exp(sum(l)) - 1."""
    check_returns(log)
    return np.exp(log.cumsum()) - 1
