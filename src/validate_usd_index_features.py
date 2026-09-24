from pathlib import Path
import pandas as pd
import numpy as np

FILE = Path("data/processed/usd_index_m1_features.csv")

FEATURES = [
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "volatility_15m",
    "volatility_60m",
    "distance_sma_60",
    "distance_sma_240",
]

REQUIRED = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
] + FEATURES

CHUNK_SIZE = 100_000

print("=" * 70)
print("USD INDEX FEATURE DATASET VALIDATION")
print("=" * 70)
print(f"File: {FILE}")

if not FILE.exists():
    raise SystemExit("Feature file not found.")

total_rows = 0
missing = 0
duplicates = 0
invalid_ohlc = 0
nonpositive = 0
nonfinite_features = 0
backward_timestamps = 0
nonminute_forward = 0

first_ts = None
last_ts = None
previous_ts = None

reader = pd.read_csv(FILE, chunksize=CHUNK_SIZE)

first_chunk = True

for chunk_no, df in enumerate(reader, start=1):

    if first_chunk:
        missing_columns = [c for c in REQUIRED if c not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        first_chunk = False

    total_rows += len(df)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------
    missing += int(df[REQUIRED].isna().sum().sum())

    # --------------------------------------------------------
    # OHLC validity
    # --------------------------------------------------------
    invalid_ohlc += int(
        (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            |
            (df["low"] > df[["open", "close", "high"]].min(axis=1))
        ).sum()
    )

    nonpositive += int(
        (
            df[["open", "high", "low", "close"]] <= 0
        ).any(axis=1).sum()
    )

    # --------------------------------------------------------
    # Feature finite check
    # --------------------------------------------------------
    values = df[FEATURES].to_numpy(dtype=np.float64)

    nonfinite_features += int(
        (~np.isfinite(values)).sum()
    )

    # --------------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------------
    ts = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    if first_chunk:
        first_ts = ts.iloc[0]

    if previous_ts is not None:

        # Check boundary between chunks
        delta = ts.iloc[0] - previous_ts

        if delta <= pd.Timedelta(0):
            backward_timestamps += 1
        elif delta != pd.Timedelta(minutes=1):
            nonminute_forward += 1

    # Within-chunk timestamp checks
    diffs = ts.diff().dropna()

    backward_timestamps += int(
        (diffs <= pd.Timedelta(0)).sum()
    )

    nonminute_forward += int(
        (diffs != pd.Timedelta(minutes=1)).sum()
    )

    duplicates += int(
        df["timestamp"].duplicated().sum()
    )

    previous_ts = ts.iloc[-1]
    last_ts = ts.iloc[-1]

    if chunk_no % 5 == 0:
        print(
            f"Processed {total_rows:,} rows..."
        )

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Rows:                {total_rows:,}")
print(f"Range:               {first_ts} -> {last_ts}")
print(f"Missing values:      {missing:,}")
print(f"Duplicate timestamps:{duplicates:,}")
print(f"Invalid OHLC:        {invalid_ohlc:,}")
print(f"Non-positive prices: {nonpositive:,}")
print(f"Non-finite features: {nonfinite_features:,}")
print(f"Backward timestamps: {backward_timestamps:,}")
print(f"Non-1m transitions:  {nonminute_forward:,}")

status = (
    missing == 0
    and duplicates == 0
    and invalid_ohlc == 0
    and nonpositive == 0
    and nonfinite_features == 0
    and backward_timestamps == 0
)

print(
    f"\nFEATURE DATASET VALIDATION: "
    f"{'PASS' if status else 'FAIL'}"
)
