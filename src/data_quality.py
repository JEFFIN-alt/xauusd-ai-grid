import pandas as pd
from pathlib import Path

FILE = Path("data/raw/download/xauusd-m1-bid-2019-01-01-2019-02-01.csv")

df = pd.read_csv(FILE)

print("=" * 60)
print("XAUUSD M1 DATA QUALITY REPORT")
print("=" * 60)

print(f"\nRows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Convert timestamp
df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

print(f"\nFirst timestamp: {df['datetime'].iloc[0]}")
print(f"Last timestamp:  {df['datetime'].iloc[-1]}")

# Missing values
print("\nMissing values:")
print(df.isna().sum())

# Duplicate timestamps
duplicate_timestamps = df["timestamp"].duplicated().sum()
print(f"\nDuplicate timestamps: {duplicate_timestamps:,}")

# Chronological order
is_sorted = df["timestamp"].is_monotonic_increasing
print(f"Chronological order: {is_sorted}")

# OHLC sanity checks
bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()

print(f"\nInvalid HIGH rows: {bad_high:,}")
print(f"Invalid LOW rows:  {bad_low:,}")

# Negative / zero prices
bad_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
print(f"Invalid/non-positive price rows: {bad_prices:,}")

# Time gaps
time_diff = df["datetime"].diff().dt.total_seconds() / 60
gaps = time_diff[time_diff > 1]

print(f"\nGaps greater than 1 minute: {len(gaps):,}")

if len(gaps) > 0:
    print("\nLargest gaps:")
    print(gaps.nlargest(10))

# Flat candles
flat = (
    (df["open"] == df["high"]) &
    (df["high"] == df["low"]) &
    (df["low"] == df["close"])
).sum()

print(f"\nCompletely flat candles: {flat:,}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)
