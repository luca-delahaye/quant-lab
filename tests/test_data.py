import pandas as pd
import pytest

from quantlab import data
from quantlab.data import (
    PriceDataError,
    _flatten,
    fetch_and_save,
    load_adjusted,
    load_prices,
    validate_prices,
)

# The same prices as make_prices(), as the CSV that fetch_and_save writes.
VALID_CSV = """Date,Adj Close,Close
2024-01-02,100.0,110.0
2024-01-03,101.0,111.0
2024-01-04,102.0,112.0
"""


def make_prices():
    """Three valid days. The two columns differ so a mix-up between them shows."""
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="Date")
    return pd.DataFrame(
        {"Adj Close": [100.0, 101.0, 102.0], "Close": [110.0, 111.0, 112.0]},
        index=dates,
    )


# --- validate_prices: the airlock ---------------------------------------------


def test_valid_prices_pass_unchanged():
    prices = make_prices()
    result = validate_prices(prices)
    pd.testing.assert_frame_equal(result, prices)


def test_empty_raises():
    empty = make_prices().iloc[0:0]
    with pytest.raises(PriceDataError):
        validate_prices(empty)


def test_nan_in_adj_close_raises():
    prices = make_prices()
    prices.loc[prices.index[1], "Adj Close"] = float("nan")
    with pytest.raises(PriceDataError):
        validate_prices(prices)


def test_nan_in_close_raises():
    prices = make_prices()
    prices.loc[prices.index[1], "Close"] = float("nan")
    with pytest.raises(PriceDataError):
        validate_prices(prices)


def test_zero_price_raises():
    prices = make_prices()
    prices.loc[prices.index[1], "Adj Close"] = 0.0
    with pytest.raises(PriceDataError):
        validate_prices(prices)


def test_negative_price_raises():
    prices = make_prices()
    prices.loc[prices.index[1], "Adj Close"] = -5.0
    with pytest.raises(PriceDataError):
        validate_prices(prices)


def test_duplicate_date_raises():
    prices = make_prices()
    prices.index = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-02", "2024-01-04"], name="Date"
    )
    with pytest.raises(PriceDataError):
        validate_prices(prices)


def test_unsorted_dates_are_sorted():
    shuffled = make_prices().iloc[[2, 0, 1]]
    result = validate_prices(shuffled)
    # Equal to the original: dates sorted, and each date kept its own prices.
    pd.testing.assert_frame_equal(result, make_prices())


# --- _flatten -----------------------------------------------------------------


def test_flatten_keeps_adj_close_and_close():
    # Shaped like yf.download(..., auto_adjust=False) on Day 2: Price x Ticker.
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["SPY"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [
            [100.0, 110.0, 115.0, 105.0, 108.0, 1000],
            [101.0, 111.0, 116.0, 106.0, 109.0, 2000],
            [102.0, 112.0, 117.0, 107.0, 110.0, 3000],
        ],
        index=make_prices().index,
        columns=columns,
    )
    result = _flatten(raw, "SPY")
    # Axis names ("Price") are not part of the contract; labels and values are.
    pd.testing.assert_frame_equal(result, make_prices(), check_names=False)


# --- load_prices and load_adjusted --------------------------------------------


def test_load_prices_parses_dates(tmp_path):
    path = tmp_path / "SPY.csv"
    path.write_text(VALID_CSV)
    result = load_prices(path)
    assert isinstance(result.index, pd.DatetimeIndex)
    pd.testing.assert_frame_equal(result, make_prices())


def test_load_prices_validates(tmp_path):
    path = tmp_path / "SPY.csv"
    path.write_text(VALID_CSV.replace("101.0", "-101.0"))
    with pytest.raises(PriceDataError):
        load_prices(path)


def test_load_adjusted_returns_adj_close_series(tmp_path):
    path = tmp_path / "SPY.csv"
    path.write_text(VALID_CSV)
    result = load_adjusted(path)
    assert isinstance(result, pd.Series)
    pd.testing.assert_series_equal(result, make_prices()["Adj Close"])


# --- fetch_and_save -----------------------------------------------------------


def test_fetch_skips_download_when_file_exists(tmp_path, monkeypatch):
    path = tmp_path / "SPY.csv"
    path.write_text(VALID_CSV)

    def fail_if_called(*args, **kwargs):
        pytest.fail("download_prices was called although the file exists")

    monkeypatch.setattr(data, "download_prices", fail_if_called)
    assert fetch_and_save(path) == path
    assert path.read_text() == VALID_CSV
