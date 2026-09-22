"""Fetch, validate and load daily SPY prices.

The only module that touches the network or the CSV. Everything downstream receives
prices that already passed validate_prices(), so it never has to check them again.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER = "SPY"
START = "2005-01-01"
END = "2026-01-01"  # yfinance's end is exclusive: the data ends on 2025-12-31.
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "SPY.csv"

COLUMNS = ["Adj Close", "Close"]


class PriceDataError(ValueError):
    """Price data that cannot be trusted: empty, missing, non-positive or duplicated."""


def download_prices(ticker: str = TICKER, start: str = START, end: str = END) -> pd.DataFrame:
    """Download daily prices from Yahoo: real prices plus an Adj Close column."""
    return yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)


def _flatten(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop the ticker level from yfinance's two-level columns; keep Adj Close and Close."""
    if raw.empty:
        raise PriceDataError(f"no data downloaded for {ticker}")
    return raw.xs(ticker, axis=1, level="Ticker")[COLUMNS].rename_axis(columns=None)


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Refuse untrustworthy prices (rule 4.6). The only repair is sorting the dates."""
    if prices.empty:
        raise PriceDataError("no price data")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise PriceDataError("the index must hold dates, not text")
    missing_columns = [c for c in COLUMNS if c not in prices.columns]
    if missing_columns:
        raise PriceDataError(f"missing columns: {missing_columns}")

    has_nan = prices.isna().any(axis=1)
    if has_nan.any():
        raise PriceDataError(f"missing price on {prices.index[has_nan][0].date()}")
    non_positive = (prices <= 0).any(axis=1)
    if non_positive.any():
        raise PriceDataError(f"price <= 0 on {prices.index[non_positive][0].date()}")
    duplicated = prices.index.duplicated()
    if duplicated.any():
        raise PriceDataError(f"duplicate date {prices.index[duplicated][0].date()}")

    return prices.sort_index()


def fetch_and_save(path: Path | str = CSV_PATH, force: bool = False) -> Path:
    """Download, check and save the snapshot, unless it already exists and force is False."""
    path = Path(path)
    if path.exists() and not force:
        return path
    prices = validate_prices(_flatten(download_prices(), TICKER))
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    return path


def load_prices(path: Path | str = CSV_PATH) -> pd.DataFrame:
    """Read the snapshot with real dates as the index, and validate it again."""
    prices = pd.read_csv(Path(path), index_col="Date", parse_dates=True)
    return validate_prices(prices)


def load_adjusted(path: Path | str = CSV_PATH) -> pd.Series:
    """The one door for returns.py: validated adjusted close prices."""
    return load_prices(path)["Adj Close"]
