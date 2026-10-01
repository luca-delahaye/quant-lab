import math
from statistics import NormalDist

import pandas as pd
import pytest

from quantlab.data import PriceDataError
from quantlab.returns import ReturnsError
from quantlab.risk import (
    DEFAULT_LEVEL,
    TRADING_DAYS_PER_YEAR,
    VAR_WINDOW,
    annualized_volatility,
    backtest_var,
    daily_volatility,
    drawdown_series,
    expected_shortfall,
    max_drawdown,
    rolling_var,
    var_historical,
    var_normal,
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


# --- VaR and Expected Shortfall -----------------------------------------------

# 21 returns: two losses and nineteen gains. At 95% the quantile position is
# 0.05 * (21 - 1) = 1, so "lower" lands on the second worst return.
TAIL_WORST = -0.08
TAIL_SECOND = -0.04
VAR_95 = TAIL_SECOND
ES_95 = (TAIL_WORST + TAIL_SECOND) / 2


def make_tail_returns():
    """Twenty-one returns whose 95% VaR and ES can be read off by hand."""
    values = [TAIL_WORST, TAIL_SECOND] + [0.01] * 19
    dates = pd.bdate_range("2024-01-01", periods=len(values), name="Date")
    return pd.Series(values, index=dates, name="Adj Close")


def test_var_historical_is_the_second_worst_return():
    assert var_historical(make_tail_returns()) == VAR_95


def test_var_historical_returns_an_observed_value():
    """"lower" never interpolates, so the answer is a return that actually happened."""
    returns = make_tail_returns()
    assert var_historical(returns) in set(returns)


def test_var_historical_ignores_order():
    shuffled = make_tail_returns().sample(frac=1, random_state=0)
    assert var_historical(shuffled) == VAR_95


def test_expected_shortfall_averages_the_tail():
    assert expected_shortfall(make_tail_returns()) == pytest.approx(ES_95)


def test_expected_shortfall_is_at_least_as_bad_as_var():
    returns = make_tail_returns()
    assert expected_shortfall(returns) <= var_historical(returns)


def test_default_level_is_95_percent():
    assert DEFAULT_LEVEL == 0.95
    returns = make_tail_returns()
    assert var_historical(returns, level=0.95) == var_historical(returns)


def test_var_normal_is_mean_plus_z_times_sigma():
    returns = make_returns()  # the four-value volatility example, mean 0
    z = NormalDist().inv_cdf(0.05)
    assert var_normal(returns) == pytest.approx(z * SAMPLE_DAILY)


def test_var_normal_needs_no_tail_observations():
    """Parametric VaR rests on the mean and the std, so four returns are enough."""
    assert var_normal(make_returns(), level=0.99) < 0


@pytest.mark.parametrize("function", [var_historical, expected_shortfall])
def test_an_empty_tail_raises(function):
    """At 99%, 21 returns put nothing in the tail: 21 * 0.01 < 1."""
    with pytest.raises(ReturnsError):
        function(make_tail_returns(), level=0.99)


@pytest.mark.parametrize("function", [var_historical, var_normal, expected_shortfall])
@pytest.mark.parametrize("level", [0.05, 0.4, 0.0, 1.0, 1.5])
def test_level_outside_the_allowed_range_raises(function, level):
    """0.05 is the classic slip: it means 95% but asks for a gain."""
    with pytest.raises(ReturnsError):
        function(make_tail_returns(), level=level)


@pytest.mark.parametrize("function", [var_historical, var_normal, expected_shortfall])
def test_empty_returns_raise_for_tail_measures(function):
    with pytest.raises(ReturnsError):
        function(make_tail_returns().iloc[0:0])


@pytest.mark.parametrize("function", [var_historical, var_normal, expected_shortfall])
def test_nan_in_returns_raises_for_tail_measures(function):
    broken = make_tail_returns()
    broken.iloc[1] = float("nan")
    with pytest.raises(ReturnsError):
        function(broken)


# --- backtesting the VaR ------------------------------------------------------

SHORT_WINDOW = 20  # the smallest window a 95% level allows


def as_returns(values):
    dates = pd.bdate_range("2024-01-01", periods=len(values), name="Date")
    return pd.Series(values, index=dates, name="Adj Close")


def make_backtest_returns():
    """Twenty days of history, then three tested days of which two breach."""
    return as_returns([-0.08, -0.04] + [0.01] * 18 + [-0.09, 0.02, -0.20])


def test_rolling_var_is_as_long_as_the_returns():
    returns = make_backtest_returns()
    result = rolling_var(returns, window=SHORT_WINDOW)
    assert len(result) == len(returns)
    assert result.iloc[:SHORT_WINDOW].isna().all()
    assert result.iloc[SHORT_WINDOW:].notna().all()


def test_a_crash_does_not_inflate_its_own_var():
    """Lookahead: the VaR for a day may only use returns up to the day before."""
    calm = as_returns([0.01] * 19 + [-0.03])
    crash = as_returns(list(calm) + [-0.50])
    stated_on_the_crash_day = rolling_var(crash, window=SHORT_WINDOW).iloc[-1]
    assert stated_on_the_crash_day == var_historical(calm)
    assert crash.iloc[-1] <= stated_on_the_crash_day  # and so it counts as a breach


def test_backtest_counts_breaches_out_of_sample():
    result = backtest_var(make_backtest_returns(), window=SHORT_WINDOW)
    assert result.days == 3
    assert result.breaches == 2
    assert result.expected == pytest.approx(3 * 0.05)
    assert result.rate == pytest.approx(2 / 3)
    assert result.window == SHORT_WINDOW
    assert result.level == DEFAULT_LEVEL


def test_a_return_equal_to_the_var_is_a_breach():
    returns = as_returns([0.01] * 19 + [-0.05, -0.05])
    result = backtest_var(returns, window=SHORT_WINDOW)
    assert result.days == 1
    assert result.breaches == 1


def test_days_tested_is_the_returns_minus_the_window():
    returns = make_tail_returns()  # 21 returns
    assert backtest_var(returns, window=SHORT_WINDOW).days == len(returns) - SHORT_WINDOW


def test_default_window_is_250():
    assert VAR_WINDOW == 250


@pytest.mark.parametrize("function", [rolling_var, backtest_var])
def test_window_too_small_for_the_level_raises(function):
    """At 99% a 20-day window puts nothing in the tail."""
    with pytest.raises(ReturnsError):
        function(make_backtest_returns(), window=SHORT_WINDOW, level=0.99)


@pytest.mark.parametrize("function", [rolling_var, backtest_var])
def test_window_longer_than_the_data_raises(function):
    returns = make_backtest_returns()
    with pytest.raises(ReturnsError):
        function(returns, window=len(returns))


@pytest.mark.parametrize("function", [rolling_var, backtest_var])
def test_backtest_refuses_bad_returns(function):
    broken = make_backtest_returns()
    broken.iloc[5] = float("nan")
    with pytest.raises(ReturnsError):
        function(broken, window=SHORT_WINDOW)
    with pytest.raises(ReturnsError):
        function(broken.iloc[0:0], window=SHORT_WINDOW)
