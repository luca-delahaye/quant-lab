import math

import numpy as np
import pandas as pd
import pytest

from quantlab.data import PriceDataError
from quantlab.returns import (
    ReturnsError,
    cumulative_from_log,
    cumulative_from_simple,
    log_returns,
    simple_returns,
)

DATES = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")

UP = 0.1
DOWN = -1 / 11  # (100 - 110) / 110
LOG_UP = math.log(1.1)


def make_prices():
    """100 -> 110 -> 100: the cumulative return over the whole period is exactly 0."""
    return pd.Series([100.0, 110.0, 100.0], index=DATES, name="Adj Close")


# --- the two return functions -------------------------------------------------


def test_simple_returns_on_known_prices():
    result = simple_returns(make_prices())
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(UP)
    assert result.iloc[1] == pytest.approx(DOWN)


def test_log_returns_on_known_prices():
    result = log_returns(make_prices())
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(LOG_UP)
    assert result.iloc[1] == pytest.approx(-LOG_UP)
    # Log returns add up over time: these two cancel out exactly.
    assert result.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("function", [simple_returns, log_returns])
def test_returns_drop_only_the_first_date(function):
    result = function(make_prices())
    assert len(result) == len(make_prices()) - 1
    pd.testing.assert_index_equal(result.index, DATES[1:])


# --- cumulative returns -------------------------------------------------------


def test_cumulative_from_simple_is_a_running_series():
    result = cumulative_from_simple(simple_returns(make_prices()))
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(UP)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_cumulative_from_log_is_a_running_series():
    result = cumulative_from_log(log_returns(make_prices()))
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(UP)
    assert result.iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_both_methods_agree():
    prices = make_prices()
    from_simple = cumulative_from_simple(simple_returns(prices))
    from_log = cumulative_from_log(log_returns(prices))
    pd.testing.assert_series_equal(from_simple, from_log)


def test_log_returns_rebuild_the_prices():
    """Log returns lose nothing: the prices come back from the first one."""
    prices = make_prices()
    rebuilt = prices.iloc[0] * np.exp(log_returns(prices).cumsum())
    pd.testing.assert_series_equal(rebuilt, prices.iloc[1:], check_names=False)


def test_summing_simple_returns_is_the_wrong_answer():
    """The Day 1 bug, written down: r.cumsum() gives +0.91%, not 0."""
    wrong = simple_returns(make_prices()).cumsum().iloc[-1]
    assert wrong == pytest.approx(UP + DOWN)
    assert abs(wrong) > 1e-6  # nowhere near the correct 0


# --- input checks: prices -----------------------------------------------------


@pytest.mark.parametrize("function", [simple_returns, log_returns])
def test_one_price_raises(function):
    one_day = make_prices().iloc[:1]
    with pytest.raises(PriceDataError):
        function(one_day)


@pytest.mark.parametrize("function", [simple_returns, log_returns])
def test_nan_price_raises(function):
    prices = make_prices()
    prices.iloc[1] = float("nan")
    with pytest.raises(PriceDataError):
        function(prices)


@pytest.mark.parametrize("function", [simple_returns, log_returns])
def test_zero_price_raises(function):
    prices = make_prices()
    prices.iloc[1] = 0.0
    with pytest.raises(PriceDataError):
        function(prices)


def test_negative_price_raises_for_log_returns():
    prices = make_prices()
    prices.iloc[1] = -110.0
    with pytest.raises(PriceDataError):
        log_returns(prices)


# --- input checks: returns ----------------------------------------------------


@pytest.mark.parametrize("function", [cumulative_from_simple, cumulative_from_log])
def test_empty_returns_raise(function):
    empty = simple_returns(make_prices()).iloc[0:0]
    with pytest.raises(ReturnsError):
        function(empty)


@pytest.mark.parametrize("function", [cumulative_from_simple, cumulative_from_log])
def test_nan_in_returns_raises(function):
    broken = simple_returns(make_prices())
    broken.iloc[1] = float("nan")
    with pytest.raises(ReturnsError):
        function(broken)
