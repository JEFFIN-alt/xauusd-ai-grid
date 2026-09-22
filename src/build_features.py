import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# XAUUSD MEMORY-SAFE FEATURE ENGINEERING
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_target.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_features.csv"
)

CHUNK_SIZE = 100_000
LOOKBACK = 240


print("=" * 70)
print("XAUUSD FEATURE ENGINEERING")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# HISTORY CARRIED BETWEEN CHUNKS
# ============================================================

carry = pd.DataFrame(
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "future_return_15m"
    ]
)

first_chunk = True

total_input_rows = 0
total_output_rows = 0


# ============================================================
# PROCESS DATA IN CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_input_rows += len(chunk)

    # --------------------------------------------------------
    # Add historical rows from previous chunk
    # --------------------------------------------------------

    if not carry.empty:

        working = pd.concat(
            [carry, chunk],
            ignore_index=True
        )

        carry_length = len(carry)

    else:

        working = chunk.copy()

        carry_length = 0


    # --------------------------------------------------------
    # Keep chronological order
    # --------------------------------------------------------

    working = working.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    close = working["close"]


    # ========================================================
    # RETURN FEATURES
    # ========================================================

    working["return_1m"] = (
        close.pct_change(fill_method=None)
    )

    working["return_5m"] = (
        close / close.shift(5) - 1
    )

    working["return_15m"] = (
        close / close.shift(15) - 1
    )

    working["return_30m"] = (
        close / close.shift(30) - 1
    )

    working["return_60m"] = (
        close / close.shift(60) - 1
    )


    # ========================================================
    # CANDLE FEATURES
    # ========================================================

    working["candle_range"] = (
        working["high"] - working["low"]
    )

    working["body"] = (
        working["close"] - working["open"]
    )

    working["abs_body"] = (
        working["body"].abs()
    )

    working["upper_wick"] = (
        working["high"]
        - working[["open", "close"]].max(axis=1)
    )

    working["lower_wick"] = (
        working[["open", "close"]].min(axis=1)
        - working["low"]
    )


    # ========================================================
    # CANDLE RATIOS
    # ========================================================

    safe_range = working[
        "candle_range"
    ].replace(0, np.nan)


    working["body_range_ratio"] = (
        working["abs_body"] / safe_range
    )

    working["upper_wick_ratio"] = (
        working["upper_wick"] / safe_range
    )

    working["lower_wick_ratio"] = (
        working["lower_wick"] / safe_range
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Flat candles are REAL observations.
    #
    # When high == low == open == close,
    # candle range = 0.
    #
    # Instead of deleting these rows, define their
    # candle ratios as zero.
    # --------------------------------------------------------

    flat_candle = (
        working["candle_range"] == 0
    )


    working.loc[
        flat_candle,
        [
            "body_range_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio"
        ]
    ] = 0.0


    # ========================================================
    # VOLATILITY
    # ========================================================

    working["volatility_15m"] = (
        working["return_1m"].rolling(15).std()
    )

    working["volatility_30m"] = (
        working["return_1m"].rolling(30).std()
    )

    working["volatility_60m"] = (
        working["return_1m"].rolling(60).std()
    )

    working["volatility_240m"] = (
        working["return_1m"].rolling(240).std()
    )


    # ========================================================
    # MOMENTUM
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# XAUUSD MEMORY-SAFE FEATURE ENGINEERING
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_target.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_features.csv"
)

CHUNK_SIZE = 100_000
LOOKBACK = 240


print("=" * 70)
print("XAUUSD FEATURE ENGINEERING")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# HISTORY CARRIED BETWEEN CHUNKS
# ============================================================

carry = pd.DataFrame(
    columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "future_return_15m"
    ]
)

first_chunk = True

total_input_rows = 0
total_output_rows = 0


# ============================================================
# PROCESS DATA IN CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    total_input_rows += len(chunk)

    # --------------------------------------------------------
    # Add historical rows from previous chunk
    # --------------------------------------------------------

    if not carry.empty:

        working = pd.concat(
            [carry, chunk],
            ignore_index=True
        )

        carry_length = len(carry)

    else:

        working = chunk.copy()

        carry_length = 0


    # --------------------------------------------------------
    # Keep chronological order
    # --------------------------------------------------------

    working = working.sort_values(
        "timestamp"
    ).reset_index(drop=True)


    close = working["close"]


    # ========================================================
    # RETURN FEATURES
    # ========================================================

    working["return_1m"] = (
        close.pct_change(fill_method=None)
    )

    working["return_5m"] = (
        close / close.shift(5) - 1
    )

    working["return_15m"] = (
        close / close.shift(15) - 1
    )

    working["return_30m"] = (
        close / close.shift(30) - 1
    )

    working["return_60m"] = (
        close / close.shift(60) - 1
    )


    # ========================================================
    # CANDLE FEATURES
    # ========================================================

    working["candle_range"] = (
        working["high"] - working["low"]
    )

    working["body"] = (
        working["close"] - working["open"]
    )

    working["abs_body"] = (
        working["body"].abs()
    )

    working["upper_wick"] = (
        working["high"]
        - working[["open", "close"]].max(axis=1)
    )

    working["lower_wick"] = (
        working[["open", "close"]].min(axis=1)
        - working["low"]
    )


    # ========================================================
    # CANDLE RATIOS
    # ========================================================

    safe_range = working[
        "candle_range"
    ].replace(0, np.nan)


    working["body_range_ratio"] = (
        working["abs_body"] / safe_range
    )

    working["upper_wick_ratio"] = (
        working["upper_wick"] / safe_range
    )

    working["lower_wick_ratio"] = (
        working["lower_wick"] / safe_range
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Flat candles are REAL observations.
    #
    # When high == low == open == close,
    # candle range = 0.
    #
    # Instead of deleting these rows, define their
    # candle ratios as zero.
    # --------------------------------------------------------

    flat_candle = (
        working["candle_range"] == 0
    )


    working.loc[
        flat_candle,
        [
            "body_range_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio"
        ]
    ] = 0.0


    # ========================================================
    # VOLATILITY
    # ========================================================

    working["volatility_15m"] = (
        working["return_1m"].rolling(15).std()
    )

    working["volatility_30m"] = (
        working["return_1m"].rolling(30).std()
    )

    working["volatility_60m"] = (
        working["return_1m"].rolling(60).std()
    )

    working["volatility_240m"] = (
        working["return_1m"].rolling(240).std()
    )


    # ========================================================
    # MOMENTUM
    # ========================================================

    working["momentum_5m"] = (
        close - close.shift(5)
    )

    working["momentum_15m"] = (
        close - close.shift(15)
    )

    working["momentum_30m"] = (
        close - close.shift(30)
    )

    working["momentum_60m"] = (
        close - close.shift(60)
    )


    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    working["sma_15"] = (
        close.rolling(15).mean()
    )

    working["sma_60"] = (
        close.rolling(60).mean()
    )

    working["sma_240"] = (
        close.rolling(240).mean()
    )


    # ========================================================
    # DISTANCE FROM MOVING AVERAGES
    # ========================================================

    working["distance_sma_15"] = (
        close / working["sma_15"] - 1
    )

    working["distance_sma_60"] = (
        close / working["sma_60"] - 1
    )

    working["distance_sma_240"] = (
        close / working["sma_240"] - 1
    )


    # ========================================================
    # SMA SLOPES
    # ========================================================

    working["sma_15_slope"] = (
        working["sma_15"]
        - working["sma_15"].shift(15)
    )

    working["sma_60_slope"] = (
        working["sma_60"]
        - working["sma_60"].shift(60)
    )

    working["sma_240_slope"] = (
        working["sma_240"]
        - working["sma_240"].shift(240)
    )


    # ========================================================
    # ROLLING HIGH / LOW
    # ========================================================

    working["rolling_high_60"] = (
        working["high"].rolling(60).max()
    )

    working["rolling_low_60"] = (
        working["low"].rolling(60).min()
    )

    working["rolling_high_240"] = (
        working["high"].rolling(240).max()
    )

    working["rolling_low_240"] = (
        working["low"].rolling(240).min()
    )


    # ========================================================
    # DISTANCE FROM HIGH / LOW
    # ========================================================

    working["distance_high_60"] = (
        close / working["rolling_high_60"] - 1
    )

    working["distance_low_60"] = (
        close / working["rolling_low_60"] - 1
    )

    working["distance_high_240"] = (
        close / working["rolling_high_240"] - 1
    )

    working["distance_low_240"] = (
        close / working["rolling_low_240"] - 1
    )


    # ========================================================
    # TIME FEATURES
    # ========================================================

    datetime = pd.to_datetime(
        working["timestamp"],
        unit="ms",
        utc=True
    )

    hour = datetime.dt.hour
    minute = datetime.dt.minute


    working["hour_sin"] = (
        np.sin(
            2 * np.pi * hour / 24
        )
    )

    working["hour_cos"] = (
        np.cos(
            2 * np.pi * hour / 24
        )
    )

    working["minute_sin"] = (
        np.sin(
            2 * np.pi * minute / 60
        )
    )

    working["minute_cos"] = (
        np.cos(
            2 * np.pi * minute / 60
        )
    )


    # ========================================================
    # FEATURE LIST
    # ========================================================

    feature_columns = [

        "return_1m",
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",

        "candle_range",
        "body",
        "abs_body",
        "upper_wick",
        "lower_wick",

        "body_range_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",

        "volatility_15m",
        "volatility_30m",
        "volatility_60m",
        "volatility_240m",

        "momentum_5m",
        "momentum_15m",
        "momentum_30m",
        "momentum_60m",

        "sma_15",
        "sma_60",
        "sma_240",

        "distance_sma_15",
        "distance_sma_60",
        "distance_sma_240",

        "sma_15_slope",
        "sma_60_slope",
        "sma_240_slope",

        "rolling_high_60",
        "rolling_low_60",
        "rolling_high_240",
        "rolling_low_240",

        "distance_high_60",
        "distance_low_60",
        "distance_high_240",
        "distance_low_240",

        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos"
    ]


    # ========================================================
    # KEEP ONLY CURRENT CHUNK
    # ========================================================

    start = carry_length
    end = start + len(chunk)

    current = working.iloc[
        start:end
    ].copy()


    # ========================================================
    # REMOVE ONLY ROWS THAT TRULY LACK FEATURE HISTORY
    # ========================================================

    current = current.dropna(
        subset=feature_columns
    )


    # ========================================================
    # OUTPUT COLUMNS
    # ========================================================

    output_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close"
    ]

    output_columns += feature_columns

    output_columns += [
        "future_return_15m"
    ]


    current = current[
        output_columns
    ]


    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    if not current.empty:

        current.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

        total_output_rows += len(current)


    # ========================================================
    # KEEP HISTORY FOR NEXT CHUNK
    # ========================================================

    carry = working.tail(
        LOOKBACK
    )[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "future_return_15m"
        ]
    ].copy()


    print(
        f"Processed chunk {chunk_number} | "
        f"Input: {total_input_rows:,} | "
        f"Output: {total_output_rows:,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(
    f"\nInput rows:  {total_input_rows:,}"
)

print(
    f"Output rows: {total_output_rows:,}"
)

print(
    f"\nSaved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nNumber of feature columns:",
    len(feature_columns)
)

print("\n" + "=" * 70)
