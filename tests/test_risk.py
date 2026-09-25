import math

import pandas as pd
import pytest

from quantlab.data import PriceDataError
from quantlab.returns import ReturnsError
from quantlab.risk import (
    TRADING_DAYS_PER_YEAR,
    annualized_volatility,
    daily_volatility,
    drawdown_series,
    max_drawdown,
)

DATES = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], name="Date"
)
PRICE_DATES = DATES.append(pd.DatetimeIndex(["2024-01-08"], name="Date"))

# +1%, -1%, +2%, -2%: mean 0, squared deviations 0.0001 + 0.0001 + 0.0004 + 0.0004.
SQUARED_DEVIATIONS = 0.001
SAMPLE_DAILY = math.sqrt(SQUARED_DEVIATIONS / 3)  # ddof=1, divide by N - 1
POPULATION_DAILY = math.sqrt(SQUARED_DEVIATIONS / 4)  # ddof=0, the convention we rejected


def make_returns():
    """Four returns whose standard deviation can be computed by hand."""
    return pd.Series([0.01, -0.01, 0.02, -0.02], index=DATES, name="Adj Close")


# --- the two volatility functions ---------------------------------------------


def test_daily_volatility_on_hand_computed_returns():
    assert daily_volatility(make_returns()) == pytest.approx(SAMPLE_DAILY)


def test_annualized_volatility_scales_by_sqrt_252():
    expected = SAMPLE_DAILY * math.sqrt(252)
    assert annualized_volatility(make_returns()) == pytest.approx(expected)


def test_trading_days_per_year_is_252():
    assert TRADING_DAYS_PER_YEAR == 252


def test_sample_std_not_population_std():
    """Decision 4.3 as a fact: ddof=1, so the population value must not come out."""
    assert daily_volatility(make_returns()) != pytest.approx(POPULATION_DAILY)


def test_periods_per_year_is_a_real_parameter():
    """Weekly data: nothing is hard-coded to 252."""
    expected = SAMPLE_DAILY * math.sqrt(52)
    assert annualized_volatility(make_returns(), periods_per_year=52) == pytest.approx(
        expected
    )


def test_constant_returns_have_zero_volatility():
    constant = pd.Series([0.01, 0.01, 0.01, 0.01], index=DATES)
    assert daily_volatility(constant) == 0.0
    assert annualized_volatility(constant) == 0.0


def test_order_does_not_matter():
    """Volatility measures spread, not the path taken."""
    shuffled = make_returns().iloc[[2, 0, 3, 1]]
    assert daily_volatility(shuffled) == pytest.approx(daily_volatility(make_returns()))


def test_volatility_is_a_float():
    assert isinstance(daily_volatility(make_returns()), float)
    assert isinstance(annualized_volatility(make_returns()), float)


# --- input checks -------------------------------------------------------------


@pytest.mark.parametrize("function", [daily_volatility, annualized_volatility])
def test_one_return_raises(function):
    """With ddof=1 a single return divides by zero; pandas returns NaN silently."""
    one = make_returns().iloc[:1]
    with pytest.raises(ReturnsError):
        function(one)


@pytest.mark.parametrize("function", [daily_volatility, annualized_volatility])
def test_empty_returns_raise(function):
    empty = make_returns().iloc[0:0]
    with pytest.raises(ReturnsError):
        function(empty)


@pytest.mark.parametrize("function", [daily_volatility, annualized_volatility])
def test_nan_in_returns_raises(function):
    broken = make_returns()
    broken.iloc[1] = float("nan")
    with pytest.raises(ReturnsError):
        function(broken)


# --- drawdown -----------------------------------------------------------------

# 100 -> 120 -> 90 -> 110 -> 130, so the running peaks are 100, 120, 120, 120, 130.
WORST_DRAWDOWN = -0.25  # 90 against a peak of 120
PARTIAL_RECOVERY = -1 / 12  # 110 against a peak of 120


def make_drawdown_prices():
    """Five prices whose drawdowns can be computed by hand."""
    return pd.Series(
        [100.0, 120.0, 90.0, 110.0, 130.0], index=PRICE_DATES, name="Adj Close"
    )


def test_max_drawdown_on_hand_computed_prices():
    assert max_drawdown(make_drawdown_prices()) == WORST_DRAWDOWN


def test_rising_prices_have_no_drawdown():
    """The August test: a series that only rises never falls from a peak."""
    rising = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0], index=PRICE_DATES)
    assert max_drawdown(rising) == 0.0


def test_halved_prices_give_minus_one_half():
    halved = pd.Series([100.0, 50.0], index=PRICE_DATES[:2])
    assert max_drawdown(halved) == -0.5


def test_drawdown_is_scale_invariant():
    """A drawdown is a ratio: multiplying every price by a constant changes nothing."""
    prices = make_drawdown_prices()
    assert max_drawdown(prices * 3) == pytest.approx(max_drawdown(prices))


def test_drawdown_series_matches_the_prices():
    expected = pd.Series(
        [0.0, 0.0, WORST_DRAWDOWN, PARTIAL_RECOVERY, 0.0],
        index=PRICE_DATES,
        name="Adj Close",
    )
    pd.testing.assert_series_equal(drawdown_series(make_drawdown_prices()), expected)


def test_drawdown_is_never_positive():
    """Decision 4.4 as a fact: negative below a peak, exactly 0 at a new high."""
    assert (drawdown_series(make_drawdown_prices()) <= 0).all()
    assert max_drawdown(make_drawdown_prices()) <= 0


def test_one_price_has_zero_drawdown():
    """Unlike volatility, a single price is answerable: it is its own peak."""
    one = make_drawdown_prices().iloc[:1]
    assert max_drawdown(one) == 0.0
    assert (drawdown_series(one) == 0.0).all()


@pytest.mark.parametrize("function", [drawdown_series, max_drawdown])
def test_empty_prices_raise(function):
    with pytest.raises(PriceDataError):
        function(make_drawdown_prices().iloc[0:0])


@pytest.mark.parametrize("function", [drawdown_series, max_drawdown])
@pytest.mark.parametrize("bad", [float("nan"), 0.0, -90.0])
def test_bad_price_raises(function, bad):
    prices = make_drawdown_prices()
    prices.iloc[2] = bad
    with pytest.raises(PriceDataError):
        function(prices)
