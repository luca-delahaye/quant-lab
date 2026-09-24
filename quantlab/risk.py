"""Risk measures computed from returns.

Volatility today; VaR and Expected Shortfall in week 2. Conventions (4.3, 4.5):
sample standard deviation with ddof=1, annualized with sqrt(252).
"""

import math

import pandas as pd

from quantlab.returns import ReturnsError, check_returns

TRADING_DAYS_PER_YEAR = 252

MIN_RETURNS = 2  # ddof=1 divides by N - 1, so one return says nothing about spread.


def daily_volatility(returns: pd.Series) -> float:
    """Standard deviation of the returns, ddof=1 written out (4.3)."""
    check_returns(returns)
    if len(returns) < MIN_RETURNS:
        raise ReturnsError(
            f"at least {MIN_RETURNS} returns are needed to measure spread, got {len(returns)}"
        )
    return float(returns.std(ddof=1))


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Daily volatility scaled to a year: variance adds over time, so volatility scales by sqrt.

    periods_per_year = 52 for weekly returns, 12 for monthly ones.
    """
    return daily_volatility(returns) * math.sqrt(periods_per_year)
