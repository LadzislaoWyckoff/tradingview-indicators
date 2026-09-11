"""Data loading, the diagnostics battery, and a CLI smoke test."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from garchlab.cli import main
from garchlab.data import load_csv, load_prices, to_returns
from garchlab.diagnostics import arch_lm, diagnose, jarque_bera, ljung_box, news_impact_curve
from garchlab.models import fit, simulate

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_spx.csv"


@pytest.fixture()
def csv_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(400) * 0.01))
    frame = pd.DataFrame({
        "Date": dates, "Open": close * 0.999, "High": close * 1.008,
        "Low": close * 0.992, "Close": close, "Adj Close": close * 0.98,
        "Volume": 1e6,
    })
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    return path


def test_load_csv_normalizes_columns_and_index(csv_file):
    frame = load_csv(csv_file)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing


def test_adjustment_scales_the_whole_bar_so_it_stays_consistent(csv_file):
    frame = load_csv(csv_file)
    assert (frame["low"] <= frame["close"]).all()
    assert (frame["close"] <= frame["high"]).all()
    assert (frame["low"] <= frame["open"]).all()
    assert (frame["open"] <= frame["high"]).all()


def test_date_filtering(csv_file):
    frame = load_prices(csv=csv_file, start="2020-06-01", end="2020-09-01")
    assert frame.index.min() >= pd.Timestamp("2020-06-01")
    assert frame.index.max() <= pd.Timestamp("2020-09-01")
    with pytest.raises(ValueError, match="no rows left"):
        load_prices(csv=csv_file, start="2050-01-01")


def test_log_returns_add_up_across_time(csv_file):
    frame = load_csv(csv_file)
    r = to_returns(frame, kind="log")
    total = np.log(frame["close"].iloc[-1] / frame["close"].iloc[0])
    assert float(r.sum()) == pytest.approx(total)
    assert not r.isna().any()


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")


# --- diagnostics -----------------------------------------------------------
def test_ljung_box_finds_autocorrelation_and_clears_white_noise():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(3000)
    assert ljung_box(white, 20).pvalue > 0.05
    correlated = np.empty(3000)
    correlated[0] = rng.standard_normal()
    for t in range(1, 3000):
        correlated[t] = 0.6 * correlated[t - 1] + rng.standard_normal()
    assert ljung_box(correlated, 20).pvalue < 1e-6


def test_arch_lm_finds_volatility_clustering_and_clears_its_absence():
    rng = np.random.default_rng(1)
    assert arch_lm(rng.standard_normal(3000), 12).pvalue > 0.05
    clustered = simulate(3000, "garch", [0.05, 0.15, 0.80], seed=2)[0]
    assert arch_lm(clustered, 12).pvalue < 1e-6


def test_jarque_bera_separates_normal_from_fat_tailed():
    rng = np.random.default_rng(3)
    assert jarque_bera(rng.standard_normal(4000)).pvalue > 0.05
    assert jarque_bera(rng.standard_t(3, 4000)).pvalue < 1e-6


def test_diagnose_clears_a_correctly_specified_model():
    r = simulate(4000, "gjr", [0.03, 0.02, 0.12, 0.90], seed=5)[0]
    res = fit(r, model="gjr", dist="normal", restarts=0, robust_errors=False)
    tests = diagnose(res)
    assert tests["ljung_box_z2"].pvalue > 0.01
    assert tests["arch_lm"].pvalue > 0.01
    assert tests["sign_bias"].pvalue > 0.01
    assert "z^2" in tests["ljung_box_z2"].name


def test_news_impact_curve_is_tilted_for_gjr_and_symmetric_for_garch():
    r = simulate(3000, "gjr", [0.03, 0.02, 0.12, 0.90], seed=6)[0]
    asym = fit(r, model="gjr", dist="normal", restarts=0, robust_errors=False)
    shock, vol = news_impact_curve(asym)
    down = float(np.interp(-0.03, shock, vol))
    up = float(np.interp(0.03, shock, vol))
    assert down > up * 1.05

    sym = fit(r, model="garch", dist="normal", restarts=0, robust_errors=False)
    shock, vol = news_impact_curve(sym)
    assert float(np.interp(-0.03, shock, vol)) == pytest.approx(
        float(np.interp(0.03, shock, vol)), rel=1e-6
    )


# --- CLI -------------------------------------------------------------------
@pytest.mark.skipif(not SAMPLE.exists(), reason="run examples/make_sample_data.py first")
def test_cli_fit_runs(capsys):
    code = main(["fit", "--csv", str(SAMPLE), "--start", "2018-01-01",
                 "--model", "garch", "--dist", "normal"])
    assert code == 0
    out = capsys.readouterr().out
    assert "persistence" in out and "ARCH-LM" in out


@pytest.mark.skipif(not SAMPLE.exists(), reason="run examples/make_sample_data.py first")
def test_cli_forecast_writes_json(tmp_path, capsys):
    target = tmp_path / "fc.json"
    code = main(["forecast", "--csv", str(SAMPLE), "--start", "2018-01-01",
                 "--model", "garch", "--dist", "normal",
                 "--horizon", "10", "--json", str(target)])
    assert code == 0
    payload = json.loads(target.read_text())
    assert len(payload["vol"]) == 10
    assert payload["persistence"] < 1.0
    assert 0.0 < payload["current_vol"] < 2.0


def test_cli_reports_a_missing_file_without_a_traceback(capsys):
    assert main(["fit", "--csv", "/nonexistent/prices.csv"]) == 1
    assert "error:" in capsys.readouterr().err


def test_module_entry_point_exposes_help():
    proc = subprocess.run(
        [sys.executable, "-m", "garchlab", "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert proc.returncode == 0
    assert "backtest" in proc.stdout
