"""Rule-based Japanese candlestick pattern detection and follow-through checks.

Rules are transparent approximations of common textbook definitions. They are
not a claim that a pattern predicts returns; all results are measured out of
sample on later bars and are descriptive, not a trading recommendation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_GUIDE: dict[str, str] = {
    "Doji": "Open and close are nearly equal; it signals indecision, not direction.",
    "Long-legged doji": "A doji with long shadows on both sides, indicating a wide intraday range and indecision.",
    "Dragonfly doji": "Nearly equal open/close near the high, with a long lower shadow.",
    "Gravestone doji": "Nearly equal open/close near the low, with a long upper shadow.",
    "Hammer": "Small body near the high and a long lower shadow after a decline; proposed bullish reversal.",
    "Hanging man": "Hammer-shaped candle after a rise; proposed bearish warning.",
    "Inverted hammer": "Small body near the low and long upper shadow after a decline; proposed bullish reversal.",
    "Shooting star": "Inverted-hammer shape after a rise; proposed bearish reversal.",
    "Spinning top": "A small real body with meaningful shadows on both sides; direction is unclear.",
    "Bullish marubozu": "A large up body with very short shadows, showing a strong session in one direction.",
    "Bearish marubozu": "A large down body with very short shadows, showing a strong session in one direction.",
    "Bullish engulfing": "A down candle followed by an up candle whose body covers the prior body.",
    "Bearish engulfing": "An up candle followed by a down candle whose body covers the prior body.",
    "Bullish harami": "A small up body sits inside the prior down body.",
    "Bearish harami": "A small down body sits inside the prior up body.",
    "Piercing line": "An up candle closes above the midpoint of the prior down body, without engulfing it.",
    "Dark cloud cover": "A down candle closes below the midpoint of the prior up body, without engulfing it.",
    "Tweezer bottom": "Two adjacent candles have lows within 0.1% of one another after a decline.",
    "Tweezer top": "Two adjacent candles have highs within 0.1% of one another after a rise.",
    "Morning star": "Three-candle decline, small middle body, then a strong up body through the first body's midpoint.",
    "Evening star": "Three-candle rise, small middle body, then a strong down body through the first body's midpoint.",
    "Three white soldiers": "Three consecutive up candles with progressively higher closes.",
    "Three black crows": "Three consecutive down candles with progressively lower closes.",
    "Three inside up": "A bullish harami followed by an up close above the first candle's open.",
    "Three inside down": "A bearish harami followed by a down close below the first candle's open.",
    "Bullish kicker": "A down candle followed by a strong up candle opening above the prior high.",
    "Bearish kicker": "An up candle followed by a strong down candle opening below the prior low.",
    "Bullish belt hold": "A long up candle opening near its low and closing near its high.",
    "Bearish belt hold": "A long down candle opening near its high and closing near its low.",
    "Inside bar": "The day's high and low both remain inside the previous day's range; direction is neutral.",
}


def detect_patterns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Return date, pattern, bias, and rule explanation for detected patterns."""
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(ohlc.columns):
        raise ValueError(f"OHLC data must contain {sorted(required)}.")
    df = ohlc[["Open", "High", "Low", "Close"]].copy().sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.empty or df.isna().any().any() or (df <= 0).any().any():
        raise ValueError("OHLC data must be non-empty, finite, and positive.")
    if (df["High"] < df[["Open", "Low", "Close"]].max(axis=1)).any() or (df["Low"] > df[["Open", "High", "Close"]].min(axis=1)).any():
        raise ValueError("OHLC candles have inconsistent high or low values.")
    o, h, l, c = (df[name].astype(float) for name in ("Open", "High", "Low", "Close"))
    body = (c - o).abs()
    span = (h - l).clip(lower=np.finfo(float).eps)
    bullish, bearish = c > o, c < o
    small = body <= span * 0.30
    long_lower = (pd.concat([o, c], axis=1).min(axis=1) - l) >= body * 2
    long_upper = (h - pd.concat([o, c], axis=1).max(axis=1)) >= body * 2
    flat = body <= span * 0.08
    trend_up = c.shift(1) > c.shift(6)
    trend_down = c.shift(1) < c.shift(6)
    signals: dict[str, tuple[pd.Series, str]] = {
        "Doji": (flat, "neutral"),
        "Long-legged doji": (flat & ((h - pd.concat([o, c], axis=1).max(axis=1)) > span * .3) & ((pd.concat([o, c], axis=1).min(axis=1) - l) > span * .3), "neutral"),
        "Dragonfly doji": (flat & (h - pd.concat([o, c], axis=1).max(axis=1) <= span * .1) & long_lower, "neutral"),
        "Gravestone doji": (flat & (pd.concat([o, c], axis=1).min(axis=1) - l <= span * .1) & long_upper, "neutral"),
        "Hammer": (small & long_lower & trend_down, "bullish"),
        "Hanging man": (small & long_lower & trend_up, "bearish"),
        "Inverted hammer": (small & long_upper & trend_down, "bullish"),
        "Shooting star": (small & long_upper & trend_up, "bearish"),
        "Spinning top": (small & (body > span * .08) & (long_lower | long_upper), "neutral"),
        "Bullish marubozu": (bullish & (body >= span * .9), "bullish"),
        "Bearish marubozu": (bearish & (body >= span * .9), "bearish"),
    }
    po, pc = o.shift(1), c.shift(1)
    prev_top, prev_bottom = pd.concat([po, pc], axis=1).max(axis=1), pd.concat([po, pc], axis=1).min(axis=1)
    top, bottom = pd.concat([o, c], axis=1).max(axis=1), pd.concat([o, c], axis=1).min(axis=1)
    signals.update({
        "Bullish engulfing": (bearish.shift(1).fillna(False) & bullish & (bottom <= prev_bottom) & (top >= prev_top) & (body >= body.shift(1)), "bullish"),
        "Bearish engulfing": (bullish.shift(1).fillna(False) & bearish & (bottom <= prev_bottom) & (top >= prev_top) & (body >= body.shift(1)), "bearish"),
        "Bullish harami": (bearish.shift(1).fillna(False) & bullish & (top < prev_top) & (bottom > prev_bottom) & (body < body.shift(1) * .6), "bullish"),
        "Bearish harami": (bullish.shift(1).fillna(False) & bearish & (top < prev_top) & (bottom > prev_bottom) & (body < body.shift(1) * .6), "bearish"),
        "Piercing line": (bearish.shift(1).fillna(False) & bullish & (o <= pc) & (c > (po + pc) / 2) & (c < po), "bullish"),
        "Dark cloud cover": (bullish.shift(1).fillna(False) & bearish & (o >= pc) & (c < (po + pc) / 2) & (c > po), "bearish"),
        "Tweezer bottom": (trend_down & (l.sub(l.shift(1)).abs() / l < .001) & (bearish.shift(1).fillna(False) | bullish.shift(1).fillna(False)), "bullish"),
        "Tweezer top": (trend_up & (h.sub(h.shift(1)).abs() / h < .001), "bearish"),
        "Inside bar": ((h < h.shift(1)) & (l > l.shift(1)), "neutral"),
        "Bullish kicker": (bearish.shift(1).fillna(False) & bullish & (o > h.shift(1)) & (body > span * .5), "bullish"),
        "Bearish kicker": (bullish.shift(1).fillna(False) & bearish & (o < l.shift(1)) & (body > span * .5), "bearish"),
        "Bullish belt hold": (bullish & (body > span * .65) & ((o - l) / span < .08), "bullish"),
        "Bearish belt hold": (bearish & (body > span * .65) & ((h - o) / span < .08), "bearish"),
    })
    o2, c2 = o.shift(2), c.shift(2)
    body2 = (c2 - o2).abs()
    mid1 = (o2 + c2) / 2
    middle_small = body.shift(1) < body2 * .5
    signals.update({
        "Morning star": (bearish.shift(2).fillna(False) & middle_small & bullish & (c > mid1), "bullish"),
        "Evening star": (bullish.shift(2).fillna(False) & middle_small & bearish & (c < mid1), "bearish"),
        "Three white soldiers": (bullish & bullish.shift(1).fillna(False) & bullish.shift(2).fillna(False) & (c > c.shift(1)) & (c.shift(1) > c.shift(2)), "bullish"),
        "Three black crows": (bearish & bearish.shift(1).fillna(False) & bearish.shift(2).fillna(False) & (c < c.shift(1)) & (c.shift(1) < c.shift(2)), "bearish"),
        "Three inside up": (bearish.shift(2).fillna(False) & bullish.shift(1).fillna(False) & (top.shift(1) < pd.concat([o.shift(2), c.shift(2)], axis=1).max(axis=1)) & (bottom.shift(1) > pd.concat([o.shift(2), c.shift(2)], axis=1).min(axis=1)) & bullish & (c > o.shift(2)), "bullish"),
        "Three inside down": (bullish.shift(2).fillna(False) & bearish.shift(1).fillna(False) & (top.shift(1) < pd.concat([o.shift(2), c.shift(2)], axis=1).max(axis=1)) & (bottom.shift(1) > pd.concat([o.shift(2), c.shift(2)], axis=1).min(axis=1)) & bearish & (c < o.shift(2)), "bearish"),
    })
    records = []
    for name, (condition, bias) in signals.items():
        for timestamp in df.index[condition.fillna(False).to_numpy()]:
            records.append({"date": timestamp, "pattern": name, "bias": bias, "definition": PATTERN_GUIDE[name]})
    return pd.DataFrame(records, columns=["date", "pattern", "bias", "definition"])


def evaluate_patterns(ohlc: pd.DataFrame, events: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Measure close-to-close forward direction hit rate and mean return by pattern."""
    if horizon < 1:
        raise ValueError("horizon must be at least one session.")
    close = pd.to_numeric(ohlc["Close"], errors="coerce").astype(float)
    future = close.shift(-horizon).div(close).sub(1)
    rows = []
    if events.empty:
        return pd.DataFrame(columns=["pattern", "bias", "signals", "hit_rate", "mean_forward_return"])
    joined = events.copy()
    joined["forward_return"] = joined["date"].map(future)
    joined = joined.dropna(subset=["forward_return"])
    for (pattern, bias), group in joined.groupby(["pattern", "bias"], sort=True):
        values = group["forward_return"]
        hit = values > 0 if bias == "bullish" else values < 0 if bias == "bearish" else None
        rows.append({"pattern": pattern, "bias": bias, "signals": int(len(group)), "hit_rate": float(hit.mean()) if hit is not None else None, "mean_forward_return": float(values.mean())})
    return pd.DataFrame(rows, columns=["pattern", "bias", "signals", "hit_rate", "mean_forward_return"])
