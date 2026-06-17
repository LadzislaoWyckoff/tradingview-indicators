# Trading Indicators

A collection of my custom **TradingView Pine** indicators and an **MQL5** screener,
mostly built around relative strength, volume and trend analysis for the Wyckoff workflow.

## Pine (`pine/`)

| File | What it does |
|---|---|
| `Comparison.pine` | Relative-strength comparison overlay (symbol vs benchmark) |
| `Comparison_module.pine` | Reusable comparison module |
| `Comparison_sigma_blend.pine` | Comparison variant with a sigma-blended weighting |
| `supertrend.pine` | SuperTrend trend-following overlay |
| `ROCNEW.pine` | Rate-of-Change oscillator (WMA-smoothed) with divergences |
| `Pivots.pine` | Pivot / swing levels |
| `UltraVolumeExhaustion.pine` | Volume-exhaustion signal |
| `Volbubles.pine` | Volume bubbles plotted on price |
| `barcountervol.pine` | Bar / volume counter helper |

## MQL5 (`mql5/`)

| File | What it does |
|---|---|
| `MT5_MultiAsset_RS_Screener.mq5` | Scans forex / indices / commodities for relative-strength leaders in MetaTrader 5 |

## Usage

**Pine:** open [TradingView](https://www.tradingview.com/) → Pine Editor → paste a `.pine`
file → *Add to chart*.

**MQL5:** copy the `.mq5` into `MQL5/Indicators/` in your MetaTrader 5 data folder,
compile in MetaEditor, then drag it onto a chart.

## License

These are my own indicators, shared for reference. Third-party indicators are intentionally
not included here.
