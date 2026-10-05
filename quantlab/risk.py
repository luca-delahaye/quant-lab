"""Risk measures: volatility and tail risk from returns, drawdown from prices.

Conventions (4.3, 4.4, 4.5, 4.7): sample standard deviation with ddof=1, annualized
with sqrt(252), drawdown and VaR reported negative.
"""

import math
from statistics import NormalDist
from typing import NamedTuple

import pandas as pd

from quantlab.returns import ReturnsError, check_prices, check_returns

TRADING_DAYS_PER_YEAR = 252

MIN_RETURNS = 2  # ddof=1 divides by N - 1, so one return says nothing about spread.

DEFAULT_LEVEL = 0.95
MIN_LEVEL = 0.5  # below this, level=0.05 for "95%" would silently ask for a gain.
VAR_WINDOW = 250  # one trading year of history, the Basel convention.

CHI2_95 = 3.841  # chi-square, 1 degree of freedom, 5% significance.


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


def _check_level(level: float) -> None:
    """Refuse a confidence level outside [MIN_LEVEL, 1)."""
    if not MIN_LEVEL <= level < 1:
        raise ReturnsError(f"level must be between {MIN_LEVEL} and 1, got {level}")


def _check_tail(count: int, level: float) -> None:
    """Refuse a level whose tail holds no observation."""
    needed = math.ceil(1 / (1 - level))
    if count < needed:
        raise ReturnsError(f"a {level:.0%} estimate needs at least {needed} returns, got {count}")


def var_historical(returns: pd.Series, level: float = DEFAULT_LEVEL) -> float:
    """The level-th worst return, read off the sample: negative for a loss (4.7)."""
    check_returns(returns)
    _check_level(level)
    _check_tail(len(returns), level)
    return float(returns.quantile(1 - level, interpolation="lower"))


def var_normal(returns: pd.Series, level: float = DEFAULT_LEVEL) -> float:
    """VaR under a normal assumption: mean + z * std, with z from the normal quantile."""
    _check_level(level)
    z = NormalDist().inv_cdf(1 - level)
    return float(returns.mean() + z * daily_volatility(returns))


def expected_shortfall(returns: pd.Series, level: float = DEFAULT_LEVEL) -> float:
    """Average of the returns at or below the historical VaR: how bad the tail is."""
    threshold = var_historical(returns, level)
    return float(returns[returns <= threshold].mean())

class BacktestResult(NamedTuple):
    """Scorecard of a walk-forward VaR test."""

    window: int
    level: float
    days: int
    breaches: int
    expected: float
    rate: float


def _check_window(window: int, returns: pd.Series, level: float) -> None:
    """Refuse a window too small for the level, or too long to leave a day to test."""
    _check_tail(window, level)
    if window >= len(returns):
        raise ReturnsError(
            f"a window of {window} needs more returns than that, got {len(returns)}"
        )


def rolling_var(
    returns: pd.Series, window: int = VAR_WINDOW, level: float = DEFAULT_LEVEL
) -> pd.Series:
    """The VaR stated each morning from the previous `window` days; first `window` days NaN."""
    check_returns(returns)
    _check_level(level)
    _check_window(window, returns, level)
    trailing = returns.rolling(window).quantile(1 - level, interpolation="lower")
    return trailing.shift(1).rename("VaR")


def backtest_var(
    returns: pd.Series, window: int = VAR_WINDOW, level: float = DEFAULT_LEVEL
) -> BacktestResult:
    """Count the days whose return fell at or below the VaR stated that morning."""
    stated = rolling_var(returns, window, level)
    tested = stated.notna()
    days = int(tested.sum())
    breaches = int((returns[tested] <= stated[tested]).sum())
    return BacktestResult(
        window=window,
        level=level,
        days=days,
        breaches=breaches,
        expected=days * (1 - level),
        rate=breaches / days,
    )


class KupiecResult(NamedTuple):
    """Verdict of Kupiec's proportion-of-failures test."""

    statistic: float
    critical: float
    rejected: bool


def kupiec_test(result: BacktestResult) -> KupiecResult:
    """Whether the breach count is further from its promise than chance explains."""
    n, x = result.days, result.breaches
    p = 1 - result.level
    if x == 0:
        statistic = -2 * n * math.log(1 - p)
    elif x == n:
        statistic = -2 * n * math.log(p)
    else:
        rate = x / n
        statistic = -2 * (
            (n - x) * math.log(1 - p)
            + x * math.log(p)
            - (n - x) * math.log(1 - rate)
            - x * math.log(rate)
        )

    return KupiecResult(statistic=statistic, critical=CHI2_95, rejected=statistic > CHI2_95)
