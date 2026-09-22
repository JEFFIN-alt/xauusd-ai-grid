```python
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# XAUUSD FEATURE ENGINEERING
# ============================================================

INPUT_FILE = Path("data/processed/xauusd_m1_2019_2026.csv")
OUTPUT_FILE = Path("data/features/xauusd_m1_features.csv")


print("=" * 70)
print("XAUUSD FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading dataset...")

# IMPORTANT:
# Only load the columns actually needed.
# Do NOT load the existing datetime text column.
df = pd.read_csv(
    INPUT_FILE,
    usecols=["timestamp", "open", "high", "low", "close"]
)

print(f"Rows loaded: {len(df):,}")

# ------------------------------------------------------------
# Reconstruct datetime from authoritative Unix timestamp
# ------------------------------------------------------------

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

# Make sure data is chronological
df = df.sort_values("timestamp").reset_index(drop=True)

# ============================================================
# RETURN FEATURES
# ============================================================

print("\nCreating return features...")

df["return_1m"] = df["close"].pct_change(fill_method=None)

df["log_return_1m"] = np.log(
    df["close"] / df["close"].shift(1)
)

df["return_5m"] = (
    df["close"] / df["close"].shift(5) - 1
)

df["return_15m"] = (
    df["close"] / df["close"].shift(15) - 1
)

df["return_30m"] = (
    df["close"] / df["close"].shift(30) - 1
)

df["return_60m"] = (
    df["close"] / df["close"].shift(60) - 1
)

# ============================================================
# CANDLE STRUCTURE
# ============================================================

print("Creating candle features...")

df["candle_range"] = df["high"] - df["low"]

df["body"] = df["close"] - df["open"]

df["abs_body"] = df["body"].abs()

df["upper_wick"] = (
    df["high"] - df[["open", "close"]].max(axis=1)
)

df["lower_wick"] = (
    df[["open", "close"]].min(axis=1) - df["low"]
)

# Avoid division by zero
range_safe = df["candle_range"].replace(0, np.nan)

df["body_range_ratio"] = (
    df["abs_body"] / range_safe
)

df["upper_wick_ratio"] = (
    df["upper_wick"] / range_safe
)

df["lower_wick_ratio"] = (
    df["lower_wick"] / range_safe
)

# ============================================================
# VOLATILITY
# ============================================================

print("Creating volatility features...")

df["volatility_15m"] = (
    df["return_1m"].rolling(15).std()
)

df["volatility_30m"] = (
    df["return_1m"].rolling(30).std()
)

df["volatility_60m"] = (
    df["return_1m"].rolling(60).std()
)

df["volatility_240m"] = (
    df["return_1m"].rolling(240).std()
)

df["mean_range_15m"] = (
    df["candle_range"].rolling(15).mean()
)

df["mean_range_60m"] = (
    df["candle_range"].rolling(60).mean()
)

# ============================================================
# MOMENTUM
# ============================================================

print("Creating momentum features...")

df["momentum_5m"] = (
    df["close"] - df["close"].shift(5)
)

df["momentum_15m"] = (
    df["close"] - df["close"].shift(15)
)

df["momentum_30m"] = (
    df["close"] - df["close"].shift(30)
)

df["momentum_60m"] = (
    df["close"] - df["close"].shift(60)
)

# ============================================================
# MOVING AVERAGES
# ============================================================

print("Creating moving average features...")

df["sma_15"] = (
    df["close"].rolling(15).mean()
)

df["sma_60"] = (
    df["close"].rolling(60).mean()
)

df["sma_240"] = (
    df["close"].rolling(240).mean()
)

# Distance from moving averages

df["distance_sma_15"] = (
    df["close"] / df["sma_15"] - 1
)

df["distance_sma_60"] = (
    df["close"] / df["sma_60"] - 1
)

df["distance_sma_240"] = (
    df["close"] / df["sma_240"] - 1
)

# ============================================================
# SMA SLOPES
# ============================================================

df["sma_15_slope"] = (
    df["sma_15"] - df["sma_15"].shift(15)
)

df["sma_60_slope"] = (
    df["sma_60"] - df["sma_60"].shift(60)
)

df["sma_240_slope"] = (
    df["sma_240"] - df["sma_240"].shift(240)
)

# ============================================================
# ROLLING HIGH / LOW
# ============================================================

print("Creating high/low features...")

df["rolling_high_60"] = (
    df["high"].rolling(60).max()
)

df["rolling_low_60"] = (
    df["low"].rolling(60).min()
)

df["rolling_high_240"] = (
    df["high"].rolling(240).max()
)

df["rolling_low_240"] = (
    df["low"].rolling(240).min()
)

# Distance from recent highs/lows

df["distance_high_60"] = (
    df["close"] / df["rolling_high_60"] - 1
)

df["distance_low_60"] = (
    df["close"] / df["rolling_low_60"] - 1
)

df["distance_high_240"] = (
    df["close"] / df["rolling_high_240"] - 1
)

df["distance_low_240"] = (
    df["close"] / df["rolling_low_240"] - 1
)

# ============================================================
# TIME FEATURES
# ============================================================

print("Creating time features...")

df["hour"] = df["datetime"].dt.hour

df["minute"] = df["datetime"].dt.minute

# Cyclical encoding

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["minute_sin"] = np.sin(
    2 * np.pi * df["minute"] / 60
)

df["minute_cos"] = np.cos(
    2 * np.pi * df["minute"] / 60
)

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving feature dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Output: {OUTPUT_FILE}")

print("\nFirst timestamp:")
print(df["datetime"].iloc[0])

print("\nLast timestamp:")
print(df["datetime"].iloc[-1])

print("\nFeature columns:")
print(len(df.columns))

print("\nMissing values:")
print(df.isna().sum().sum())

print("\n" + "=" * 70)
```
