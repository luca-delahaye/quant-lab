# quant-lab

A small, tested risk toolkit for daily equity prices, built on 21 years of SPY (2005–2025). It fetches prices, refuses corrupted data, and computes returns, volatility, drawdown, Value at Risk and Expected Shortfall under conventions it states.

This is a research tool, not a trading strategy. No signal is generated and nothing here is a claim about future returns.

## Results

SPY, 2005-01-03 to 2025-12-31, 5,283 trading days, dividend-adjusted closes.

| Measure | Value | How it is computed |
|---|---|---|
| Total return | +735.6% (×8.36) | `exp(sum(log returns)) - 1` |
| Annualized volatility | 19.07% | sample std of log returns, `ddof=1`, × √252 |
| Max drawdown | −55.19% | closing peak 2007-10-09 → trough 2009-03-09 |
| 95% VaR, one day | −1.78% | 5th percentile of the daily log returns |
| 95% Expected Shortfall | −2.98% | average of the returns at or below the VaR |

Volatility is not a property of an asset but of a period: the same recipe gives 41.2% for 2008 alone and 6.7% for 2017. The same holds for the tail measures, more sharply still: the 95% VaR is −4.60% on 2008 alone and −0.50% on 2017.

A normal assumption is not uniformly optimistic, it is wrong in both directions depending on where you look:

| Level | Historical VaR | Normal VaR |
|---|---|---|
| 95% | −1.78% | −1.94% |
| 99% | −3.62% | −2.75% |
| 99.9% | −8.01% | −3.67% |

Real returns cluster closer to zero than a normal does, so at 95% the normal overstates the loss. Past roughly 97.5% it runs out of tail and understates it, badly: at 99.9% it says −3.7% where the data says −8.0%, and the worst day in the sample was −11.6%.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # 93 tests, no network access
```

```python
from quantlab.data import load_adjusted
from quantlab.returns import log_returns
from quantlab.risk import annualized_volatility, expected_shortfall, max_drawdown, var_historical

prices = load_adjusted()                        # the committed snapshot, validated
returns = log_returns(prices)

annualized_volatility(returns)                  # 0.1907
max_drawdown(prices)                            # -0.5519
var_historical(returns, level=0.99)             # -0.0361
expected_shortfall(returns)                     # -0.0298, at the default 95%
```

`quantlab.data.fetch_and_save(force=True)` downloads a fresh snapshot. Nothing else touches the network, and the tests never do.

## Conventions

Every one of these changes the numbers, so each is a deliberate choice.

| Choice | Decision |
|---|---|
| Price | Dividend-adjusted close, requested explicitly with `auto_adjust=False` |
| Returns | Log returns for volatility; the first date is dropped |
| Dispersion | Sample standard deviation, `ddof=1`|
| Annualization | Fixed 252 trading days, applied as √252 |
| Drawdown | Reported negative; exactly 0 for a series that only rises |
| VaR and ES | Reported negative, like any other return. A 95% VaR of −1.78% means 5% of days lose 1.78% or more; sources usually quote the same figure as a positive loss. |
| Percentile | The lower of the two neighbouring observations, never interpolated, so a VaR is always a return that actually happened |
| Horizon | One day. Nothing here scales a VaR to ten days. |
| Bad input | Refused. Only date order is corrected. |

Invalid input means an empty download, a missing value, a zero or negative price, or a duplicated date. All of it is refused in `data.py`.

## Data

`data/SPY.csv` holds both the adjusted and the traded close, **fetched 2026-09-22** with yfinance 1.7.0. The snapshot is committed on purpose: Yahoo recalculates the whole adjusted history every time SPY pays a dividend, so pinned library versions alone do not reproduce a number. The fetch script and the data it produced are both in the repo.

## Validation

The max drawdown is checked against an independent source.

| Source | Measures | Max drawdown |
|---|---|---|
| This repo | SPY, dividend-adjusted daily closes | **−55.19%** | 
| This repo | SPY, traded daily closes | −56.47% |
| [Wikipedia][wiki], from index data | S&P 500 price index, daily closes | −56.78% |

The **dates match exactly**: a closing peak on 2007-10-09 and a closing trough on 2009-03-09, the same two days the index reached 1,565.15 and 676.53.

The depths differ for two reasons, both expected:

- **Dividends cushion the fall.** An investor holding SPY through the crisis lost 55.19%, while the price alone fell 56.47%. Adjusted prices reinvest the dividends paid during the decline.
- **SPY is a fund, not the index.** Its own traded price fell 56.47% against the index's 56.78%: a fee and tracking difference of 0.31 points over 17 months.

A second figure sometimes quoted for SPY is 50.8%. That is reproducible here as −50.78%, but only from monthly closes.

[PortfoliosLab][plab] reports 55.19% for SPY, matching this repo, though its method is not stated on the page.

## What is approximate

- **√252 assumes daily returns are independent.** They are not: the best single day of these 21 years (+14.5%, 2008-10-13) sits two days from the second worst. Big moves cluster, so an annualized figure is a convention, not a measurement.
- **Adjusted prices are revised.** Every SPY dividend rescales the entire history before it, which is why the fetch date above matters.
- **A missing trading day is invisible.** The checks catch a bad value inside the data, but not a day Yahoo omitted entirely. Detecting that needs an exchange calendar, which is out of scope here.
- **The sample standard deviation is slightly biased low** on short windows (about 8% at N = 4, 1% at N = 20, negligible over 5,282 days). `ddof=1` removes the bias in the variance, not in its square root.
- **Close-to-close only.** Intraday extremes are not in the data, so drawdowns here are shallower than intraday figures. The index's intraday extremes even fall on different days than its closing ones: 1,576.09 on 2007-10-11 and 666.79 on 2009-03-06, a fall of 57.7%.
- **A historical VaR rests on few observations.** At 99% over two years it is decided by about five days, so the figure moves a lot when the window changes. The code refuses a level whose tail would hold no observation at all, but it cannot make a thin tail reliable.
- **A VaR estimated on the past describes the past.** It assumes the next day resembles the sample it was computed from, which crises are precisely the moments it does not.

## Layout

```
quantlab/
├── data.py       fetch, validate, load          no finance knowledge
├── returns.py    simple, log, cumulative        depends on data
└── risk.py       volatility, drawdown, VaR, ES   depends on returns
tests/            one test file per module, 93 tests, all offline
data/SPY.csv      the committed snapshot
```

Dependencies point one way, `data → returns → risk`. An import pointing backwards would mean a function is in the wrong file.

Tested on Python 3.14.2. `requirements.in` lists the four direct dependencies; `requirements.txt` pins all 28 installed versions exactly.

[wiki]: https://en.wikipedia.org/wiki/United_States_bear_market_of_2007%E2%80%932009
[plab]: https://portfolioslab.com/symbol/SPY
