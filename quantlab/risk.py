"""Risk measures: volatility from returns, drawdown from prices.

VaR and Expected Shortfall come in week 2. Conventions (4.3, 4.4, 4.5): sample
standard deviation with ddof=1, annualized with sqrt(252), drawdown reported negative.
"""

import math

import pandas as pd

from quantlab.returns import ReturnsError, check_prices, check_returns

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
    """Daily volatility scaled to a year by sqrt(periods_per_year)."""
    return daily_volatility(returns) * math.sqrt(periods_per_year)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Fall from the highest price reached so far, for every day: 0 or negative (4.4)."""
    check_prices(prices, min_length=1, positive_only=True)
    running_peak = prices.cummax()
    return prices / running_peak - 1


def max_drawdown(prices: pd.Series) -> float:
    """Worst fall from a peak over the whole period, as a negative fraction (4.4)."""
    return float(drawdown_series(prices).min())
