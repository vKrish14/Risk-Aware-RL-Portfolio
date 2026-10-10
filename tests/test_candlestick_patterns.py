"""Rule detector and forward-return measurement tests."""
import pandas as pd

from src.data.candlestick_patterns import detect_patterns, evaluate_patterns


def test_detects_engulfing_and_inside_bar() -> None:
    dates = pd.date_range("2024-01-01", periods=8)
    candles = pd.DataFrame({
        "Open": [12, 11, 10, 9.5, 10, 10.2, 10.1, 10.3],
        "High": [12.5, 11.3, 10.4, 10.4, 10.4, 10.4, 10.3, 10.5],
        "Low": [11.5, 10.5, 9.5, 9.4, 9.8, 10.0, 10.1, 10.2],
        "Close": [11.8, 10.7, 9.8, 10.3, 10.2, 10.1, 10.3, 10.4],
    }, index=dates)
    events = detect_patterns(candles)
    assert ((events["date"] == dates[3]) & (events["pattern"] == "Bullish engulfing")).any()
    assert ((events["date"] == dates[6]) & (events["pattern"] == "Inside bar")).any()


def test_forward_check_omits_unavailable_horizon_and_neutral_hit_rate() -> None:
    dates = pd.date_range("2024-01-01", periods=4)
    candles = pd.DataFrame({"Open": [10, 10, 10, 10], "High": [11] * 4, "Low": [9] * 4, "Close": [10, 11, 12, 13]}, index=dates)
    events = pd.DataFrame({"date": [dates[0], dates[2], dates[3]], "pattern": ["Doji"] * 3, "bias": ["neutral"] * 3})
    measured = evaluate_patterns(candles, events, horizon=1)
    assert measured.loc[0, "signals"] == 2
    assert pd.isna(measured.loc[0, "hit_rate"])
    assert measured.loc[0, "mean_forward_return"] > 0
