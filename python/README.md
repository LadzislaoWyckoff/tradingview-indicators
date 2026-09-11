# garchlab — volatility forecasting for the S&P 500

A GARCH-family volatility model with maximum-likelihood estimation, multi-step
forecasting, and — the part that actually matters — an out-of-sample
walk-forward backtest that tells you whether the forecast beats a one-line
EWMA.

Everything is implemented from `numpy` + `scipy` + `pandas`. No `arch`, no
`statsmodels`, no model you cannot read end to end.

```
python/
  garchlab/
    models.py         GARCH, GJR-GARCH, EGARCH; MLE; forecasting; simulation
    distributions.py  standardized normal / Student-t / Hansen skewed-t
    realized.py       realized-variance proxies (squared return, Parkinson, …)
    backtest.py       walk-forward engine, QLIKE/MSE, Diebold-Mariano, benchmarks
    var.py            VaR / Expected Shortfall + Kupiec & Christoffersen tests
    diagnostics.py    Ljung-Box, ARCH-LM, Engle-Ng sign bias, news impact curve
    data.py           CSV / yfinance / stooq loaders
    cli.py            command line
  examples/           sample data generator + an end-to-end study script
  tests/              90 tests, including a no-look-ahead check on the backtest
```

## Install

```bash
cd python
pip install -r requirements.txt      # numpy, scipy, pandas
# optional, only for downloading prices:
pip install yfinance
```

## Quick start

The repository ships a **synthetic** SPX-like series so every command below
runs without a network connection:

```bash
python examples/make_sample_data.py           # writes examples/sample_spx.csv
python -m garchlab fit --csv examples/sample_spx.csv
```

For real work, export daily OHLC from TradingView / your broker / yfinance
into a CSV with `Date,Open,High,Low,Close` columns and point `--csv` at it, or
let the tool download:

```bash
python -m garchlab fit --symbol ^GSPC --start 1990-01-01
```

## What the models are

Returns are `r_t = mu + e_t`, `e_t = sigma_t * z_t`, with `z_t` standardized to
zero mean and unit variance so the variance equation owns all the scale.

| model | variance equation | why you would use it |
|---|---|---|
| `garch` | `h_t = w + a e²_{t-1} + b h_{t-1}` | symmetric baseline |
| `gjr` | `h_t = w + (a + g·1[e_{t-1}<0]) e²_{t-1} + b h_{t-1}` | **the default.** A down day raises tomorrow's variance more than an up day of the same size — on an equity index `g` is usually the most significant coefficient in the model |
| `egarch` | `log h_t = w + a(\|z_{t-1}\|−E\|z\|) + g z_{t-1} + b log h_{t-1}` | asymmetry in logs, no positivity constraints needed |

Innovations: `normal`, `t` (Student-t), `skewt` (Hansen 1994 skewed-t, the
default — equity returns are both fat-tailed *and* left-skewed).

## Commands

```bash
# estimate + specification tests
python -m garchlab fit --csv examples/sample_spx.csv --model gjr --dist skewt

# forecast the volatility term structure 22 trading days out
python -m garchlab forecast --csv examples/sample_spx.csv --horizon 22 --json fc.json

# walk-forward out-of-sample backtest against EWMA / rolling std
python -m garchlab backtest --csv examples/sample_spx.csv --train 2000 --refit 63 \
    --proxy garman_klass

# VaR backtest and tomorrow's risk levels
python -m garchlab var --csv examples/sample_spx.csv --alpha 0.01 0.05

# fit all nine model × distribution pairs and rank by BIC
python -m garchlab compare --csv examples/sample_spx.csv
```

Or the whole study in one go:

```bash
python examples/spx_study.py
```

## From Python

```python
from garchlab import load_prices, to_returns, fit, walk_forward

prices = load_prices(csv="spx.csv")
r = to_returns(prices)

res = fit(r, model="gjr", dist="skewt")
print(res.summary())
print(f"{res.conditional_vol[-1]:.2%} now, reverting to {res.long_run_vol:.2%}")

fc = res.forecast(horizon=22)
fc["vol"]             # per-day annualized volatility, 22 steps
fc["cumulative_vol"]  # volatility of the h-day aggregate return, per period

bt = walk_forward(r, model="gjr", dist="skewt", train=2000, refit=63)
print(bt.summary())
```

## Reading the output

**`persistence`** — how much of today's variance shock survives to tomorrow.
Equity indices sit at 0.95–0.99. **`half-life`** is the same number in trading
days: how long a volatility spike takes to decay halfway back to normal.

**The forecast term structure** is the whole value of the model. If current
volatility is below the long-run level the forecast rises toward it, and if it
is above, it falls — at a rate set by the persistence. That is what a rolling
standard deviation cannot tell you.

**`QLIKE`** is the ranking metric in the backtest, not R². Realized variance
can only be measured with a noisy proxy, and Patton (2011) showed that QLIKE
and MSE are the losses that still rank models correctly under that noise —
most other loss functions, R² included, can rank them backwards. Lower is
better. **`MZ beta`** is the slope of `realized = a + b·forecast`; below 1
means the forecast over-reacts and shrinking it toward its own mean would
improve it.

**Diebold-Mariano** tests whether the gap between two forecasts' losses is
bigger than sampling noise. Beating EWMA on average is easy; beating it
significantly is the bar.

## What to expect

`examples/study_output.txt` is the full output of `examples/spx_study.py` on the
bundled synthetic series — 7,000 days, 4,999 out-of-sample one-day forecasts,
scored against the Garman-Klass proxy:

```
model                        QLIKE           MSE    MAE(vol)    MZ R2  MZ beta
gjr                       0.278991     1.028e-08    0.002772    0.676    0.584
ewma(0.94)                0.286780     1.326e-08    0.002892    0.601    0.527
rolling_std(60)           0.336243     1.713e-08    0.003218    0.370    0.449
unconditional             0.548773     1.453e-08    0.004239    0.003    0.742

Diebold-Mariano vs the GARCH model (QLIKE loss):
  vs ewma(0.94)         DM= -1.475  p=0.1403   GARCH better (not significant)
  vs rolling_std(60)    DM= -6.627  p=0.0000   GARCH better (significant)
  vs unconditional      DM=-11.248  p=0.0000   GARCH better (significant)
```

Read that honestly. GJR-GARCH crushes a rolling standard deviation and the
unconditional variance, and it wins on every loss against EWMA — but **the
margin over EWMA is not statistically significant at a one-day horizon**, and
this is on data generated by a GJR process, where the model is not even
misspecified. That is the well-documented result for daily equity volatility,
and any implementation that claims a large, significant edge over EWMA at one
day should be checked for look-ahead before it is believed.

Where the model does earn its keep is everything EWMA cannot do at all: a mean-
reverting *term structure* (10.6% now → 11.8% average over 22 days → 14.2%
long-run), a conditional distribution to draw quantiles from, and VaR that
passes Kupiec and Christoffersen at both 1% and 5% out of sample. The
forecast horizon is where GARCH separates from EWMA, not the next day.

## Things that are easy to get wrong, and how this handles them

**Look-ahead in the backtest.** Parameters are re-estimated only every `refit`
days, and between refits the variance recursion is carried forward on newly
observed returns with the parameters frozen — exactly what you could do live.
`tests/test_backtest.py` verifies this directly: changing the tail of the
series leaves every earlier forecast bit-identical.

**The realized-variance yardstick.** The default proxy is the squared return,
which is unbiased but so noisy that good forecasts score badly. If you have
OHLC data, pass `--proxy garman_klass` — roughly seven times more efficient,
and the same forecast will suddenly look much better because the ruler stopped
shaking.

**Optimization.** GARCH likelihoods with a fat-tailed, skewed innovation have a
degenerate corner where the shape parameter collapses toward `nu = 2`, the
variance dynamics switch off, and the tail explains everything. The fit looks
plausible and is garbage. Two guards: estimation starts from a Gaussian QMLE
pass (consistent for the variance parameters even when the innovation is not
normal), and the optimizer is never allowed to return a point worse than the
one it started from.

**Adjusted prices.** When a CSV carries `Adj Close`, the whole bar is scaled by
the same factor rather than replacing the close alone — otherwise the close
lands outside its own high/low and every range-based proxy silently breaks.

## Scale and runtime

Estimation on 7,000 daily observations takes 2–15 seconds depending on the
model and distribution. A walk-forward backtest is dominated by the refits:
7,000 observations with `--train 2000 --refit 63` is ~80 refits, a few minutes.
Use `--refit 63` (quarterly) on long samples rather than the default 21.

## Tests

```bash
cd python && python -m pytest -q
```

The estimator is validated by **parameter recovery**: simulate from a known
data generating process, estimate, and check the fit finds it back. That is the
test that catches the subtle failures — one of the tests in `test_models.py` is
a regression test for a real bug where the stationarity constraint accidentally
included the distribution's shape parameters, so the optimizer satisfied
`alpha + gamma/2 + beta + nu + lambda < 1` by switching the volatility dynamics
off entirely.

## Caveats

- `examples/sample_spx.csv` is **simulated**, not real market data. It is
  calibrated to look like SPX daily returns (≈16% annualized volatility,
  kurtosis ≈11, skew ≈−0.7, worst day ≈−11%) so the examples run offline, but
  no number derived from it says anything about the actual index.
- A volatility forecast is not a return forecast. GARCH says nothing about
  direction.
- Daily GARCH is beaten by models using intraday data (HAR-RV on realized
  variance) and, for short horizons, often by the options market's own implied
  volatility. This is the right baseline, not the frontier.

## References

- Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*
- Glosten, Jagannathan & Runkle (1993), *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*
- Nelson (1991), *Conditional Heteroskedasticity in Asset Returns: A New Approach*
- Hansen (1994), *Autoregressive Conditional Density Estimation*
- Bollerslev & Wooldridge (1992), *Quasi-Maximum Likelihood Estimation and Inference in Dynamic Models with Time-Varying Covariances*
- Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*
- Christoffersen (1998), *Evaluating Interval Forecasts*
