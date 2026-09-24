import os
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_2019_2026.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_15m_features_v2.csv"
)

CHUNK_SIZE = 100_000

TARGET_HORIZON = 15

# Largest historical dependency:
# sma_240 + sma_240 slope over another 240 periods
MAX_LOOKBACK = 480


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [

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
    "minute_cos",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def grouped_shift(series, groups, periods):
    """
    Shift within each continuous market segment.
    This prevents lag features from crossing timestamp gaps.
    """

    return (
        series
        .groupby(groups, sort=False)
        .shift(periods)
    )


def grouped_rolling_std(series, groups, window):
    """
    Rolling standard deviation within continuous segments.
    """

    result = (
        series
        .groupby(groups, sort=False)
        .rolling(
            window=window,
            min_periods=window
        )
        .std()
    )

    return result.reset_index(
        level=0,
        drop=True
    )


def grouped_rolling_mean(series, groups, window):
    """
    Rolling mean within continuous segments.
    """

    result = (
        series
        .groupby(groups, sort=False)
        .rolling(
            window=window,
            min_periods=window
        )
        .mean()
    )

    return result.reset_index(
        level=0,
        drop=True
    )


def grouped_rolling_max(series, groups, window):
    """
    Rolling maximum within continuous segments.
    """

    result = (
        series
        .groupby(groups, sort=False)
        .rolling(
            window=window,
            min_periods=window
        )
        .max()
    )

    return result.reset_index(
        level=0,
        drop=True
    )


def grouped_rolling_min(series, groups, window):
    """
    Rolling minimum within continuous segments.
    """

    result = (
        series
        .groupby(groups, sort=False)
        .rolling(
            window=window,
            min_periods=window
        )
        .min()
    )

    return result.reset_index(
        level=0,
        drop=True
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("XAUUSD GAP-AWARE FEATURE ENGINEERING V2")
print("=" * 70)

print(f"\nInput:")
print(INPUT_FILE)

print(f"\nOutput:")
print(OUTPUT_FILE)

print(
    f"\nMaximum historical dependency: "
    f"{MAX_LOOKBACK} consecutive M1 bars"
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_FILE.exists():

    OUTPUT_FILE.unlink()

    print(
        "\nRemoved existing V2 output."
    )


# ============================================================
# STATE
# ============================================================

carry = pd.DataFrame()

first_output = True

last_processed_timestamp = None

total_input_rows = 0

total_output_rows = 0

total_target_valid = 0

total_feature_valid = 0

total_gap_rows = 0


# ============================================================
# PROCESS CHUNKS
# ============================================================

for chunk_number, chunk in enumerate(

    pd.read_csv(
        INPUT_FILE,

        usecols=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ],

        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    total_input_rows += len(chunk)

    # --------------------------------------------------------
    # COMBINE HISTORY + CURRENT CHUNK
    # --------------------------------------------------------

    if not carry.empty:

        working = pd.concat(
            [
                carry,
                chunk,
            ],
            ignore_index=True
        )

    else:

        working = chunk.copy()


    # --------------------------------------------------------
    # ENSURE ORDER
    # --------------------------------------------------------

    working = working.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # GAP DETECTION
    # --------------------------------------------------------

    timestamp_diff = (
        working["timestamp"]
        .diff()
    )

    # A continuous M1 sequence requires exactly
    # 60,000 ms between observations.

    is_new_segment = (
        timestamp_diff
        != 60_000
    )

    # First row of a working buffer always starts
    # a local segment.

    is_new_segment.iloc[0] = True

    working["segment_id"] = (
        is_new_segment
        .cumsum()
        .astype("int64")
    )

    total_gap_rows += (
        is_new_segment
        .iloc[1:]
        .sum()
    )


    # --------------------------------------------------------
    # BASIC SERIES
    # --------------------------------------------------------

    close = working["close"]

    high = working["high"]

    low = working["low"]

    open_price = working["open"]

    segment = working["segment_id"]


    # ========================================================
    # BACKWARD RETURNS
    # ========================================================

    for horizon in [
        1,
        5,
        15,
        30,
        60,
    ]:

        previous_close = grouped_shift(
            close,
            segment,
            horizon
        )

        working[
            f"return_{horizon}m"
        ] = (
            close /
            previous_close
        ) - 1.0


    # ========================================================
    # CANDLE FEATURES
    # ========================================================

    working["candle_range"] = (
        high - low
    )

    working["body"] = (
        close - open_price
    )

    working["abs_body"] = (
        working["body"].abs()
    )

    working["upper_wick"] = (
        high -
        pd.concat(
            [
                open_price,
                close,
            ],
            axis=1
        ).max(axis=1)
    )

    working["lower_wick"] = (
        pd.concat(
            [
                open_price,
                close,
            ],
            axis=1
        ).min(axis=1)
        - low
    )


    # ========================================================
    # CANDLE RATIOS
    # ========================================================

    safe_range = (
        working["candle_range"]
        .replace(0, np.nan)
    )

    working["body_range_ratio"] = (
        working["abs_body"] /
        safe_range
    )

    working["upper_wick_ratio"] = (
        working["upper_wick"] /
        safe_range
    )

    working["lower_wick_ratio"] = (
        working["lower_wick"] /
        safe_range
    )


    # Flat candles are retained.

    flat = (
        working["candle_range"] == 0
    )

    working.loc[
        flat,
        [
            "body_range_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
        ]
    ] = 0.0


    # ========================================================
    # VOLATILITY
    # ========================================================

    working["volatility_15m"] = (
        grouped_rolling_std(
            working["return_1m"],
            segment,
            15
        )
    )

    working["volatility_30m"] = (
        grouped_rolling_std(
            working["return_1m"],
            segment,
            30
        )
    )

    working["volatility_60m"] = (
        grouped_rolling_std(
            working["return_1m"],
            segment,
            60
        )
    )

    working["volatility_240m"] = (
        grouped_rolling_std(
            working["return_1m"],
            segment,
            240
        )
    )


    # ========================================================
    # MOMENTUM
    # ========================================================

    for horizon in [
        5,
        15,
        30,
        60,
    ]:

        previous_close = grouped_shift(
            close,
            segment,
            horizon
        )

        working[
            f"momentum_{horizon}m"
        ] = (
            close -
            previous_close
        )


    # ========================================================
    # MOVING AVERAGES
    # ========================================================

    working["sma_15"] = (
        grouped_rolling_mean(
            close,
            segment,
            15
        )
    )

    working["sma_60"] = (
        grouped_rolling_mean(
            close,
            segment,
            60
        )
    )

    working["sma_240"] = (
        grouped_rolling_mean(
            close,
            segment,
            240
        )
    )


    # ========================================================
    # DISTANCE FROM MOVING AVERAGES
    # ========================================================

    working["distance_sma_15"] = (
        close /
        working["sma_15"]
        - 1.0
    )

    working["distance_sma_60"] = (
        close /
        working["sma_60"]
        - 1.0
    )

    working["distance_sma_240"] = (
        close /
        working["sma_240"]
        - 1.0
    )


    # ========================================================
    # SMA SLOPES
    # ========================================================

    working["sma_15_slope"] = (
        working["sma_15"]
        -
        grouped_shift(
            working["sma_15"],
            segment,
            15
        )
    )

    working["sma_60_slope"] = (
        working["sma_60"]
        -
        grouped_shift(
            working["sma_60"],
            segment,
            60
        )
    )

    working["sma_240_slope"] = (
        working["sma_240"]
        -
        grouped_shift(
            working["sma_240"],
            segment,
            240
        )
    )


    # ========================================================
    # ROLLING HIGH / LOW
    # ========================================================

    working["rolling_high_60"] = (
        grouped_rolling_max(
            high,
            segment,
            60
        )
    )

    working["rolling_low_60"] = (
        grouped_rolling_min(
            low,
            segment,
            60
        )
    )

    working["rolling_high_240"] = (
        grouped_rolling_max(
            high,
            segment,
            240
        )
    )

    working["rolling_low_240"] = (
        grouped_rolling_min(
            low,
            segment,
            240
        )
    )


    # ========================================================
    # DISTANCE FROM HIGH / LOW
    # ========================================================

    working["distance_high_60"] = (
        close /
        working["rolling_high_60"]
        - 1.0
    )

    working["distance_low_60"] = (
        close /
        working["rolling_low_60"]
        - 1.0
    )

    working["distance_high_240"] = (
        close /
        working["rolling_high_240"]
        - 1.0
    )

    working["distance_low_240"] = (
        close /
        working["rolling_low_240"]
        - 1.0
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
    # EXACT +15 MINUTE TARGET
    # ========================================================

    future_close = (
        close.shift(
            -TARGET_HORIZON
        )
    )

    future_timestamp = (
        working["timestamp"]
        .shift(
            -TARGET_HORIZON
        )
    )

    future_segment = (
        working["segment_id"]
        .shift(
            -TARGET_HORIZON
        )
    )

    future_time_diff = (
        future_timestamp -
        working["timestamp"]
    )

    expected_diff = (
        TARGET_HORIZON *
        60 *
        1000
    )

    working["future_return_15m"] = (
        future_close /
        close
        - 1.0
    )

    target_valid = (
        (future_time_diff == expected_diff)
        &
        (future_segment == working["segment_id"])
    )

    # ========================================================
    # DETERMINE WHICH ROWS ARE FINALIZED
    #
    # Last 15 rows cannot yet have their future target,
    # because future rows may exist in the next chunk.
    # ========================================================

    finalized_count = (
        len(working) -
        TARGET_HORIZON
    )

    finalized_mask = (
        np.arange(len(working))
        <
        finalized_count
    )


    # Count only rows that have not already been counted
    # from the previous chunk.

    new_target_mask = (
        finalized_mask &
        target_valid
    )

    if (
        last_processed_timestamp
        is not None
    ):

        new_target_mask &= (
            working["timestamp"].to_numpy()
            >
            last_processed_timestamp
        )

    total_target_valid += int(
        new_target_mask.sum()
    )


    if (
        last_processed_timestamp
        is not None
    ):

        finalized_mask &= (
            working["timestamp"].to_numpy()
            >
            last_processed_timestamp
        )


    # ========================================================
    # FEATURE VALIDITY
    # ========================================================

    feature_valid = (
        working[
            FEATURE_COLUMNS
        ]
        .notna()
        .all(axis=1)
    )

    final_valid = (
        finalized_mask
        &
        feature_valid
        &
        target_valid
        &
        np.isfinite(
            working[
                FEATURE_COLUMNS +
                [
                    "future_return_15m"
                ]
            ]
        ).all(axis=1)
    )


    total_feature_valid += (
        (
            finalized_mask &
            feature_valid
        ).sum()
    )


    # ========================================================
    # WRITE VALID ROWS
    # ========================================================

    current = working.loc[
        final_valid,
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
        +
        FEATURE_COLUMNS
        +
        [
            "future_return_15m"
        ]
    ].copy()


    if not current.empty:

        mode = (
            "w"
            if first_output
            else "a"
        )

        with OUTPUT_FILE.open(
            mode=mode,
            encoding="utf-8",
            newline=""
        ) as handle:

            current.to_csv(
                handle,
                header=first_output,
                index=False,
            )

            handle.flush()
            os.fsync(handle.fileno())

        first_output = False

        total_output_rows += (
            len(current)
        )


    # ========================================================
    # ADVANCE FINALIZED POSITION
    # ========================================================

    finalized_timestamps = (
        working.loc[
            finalized_mask,
            "timestamp"
        ]
    )

    if not finalized_timestamps.empty:

        last_processed_timestamp = int(
            finalized_timestamps.iloc[-1]
        )


    # ========================================================
    # KEEP HISTORY
    # ========================================================

    carry = working.tail(
        MAX_LOOKBACK
    )[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    ].copy()


    # ========================================================
    # PROGRESS
    # ========================================================

    print(
        f"Chunk {chunk_number:02d} | "
        f"Input: {total_input_rows:,} | "
        f"Output: {total_output_rows:,} | "
        f"Working rows: {len(working):,}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING V2 COMPLETE")
print("=" * 70)

print(
    f"\nInput rows:          "
    f"{total_input_rows:,}"
)

print(
    f"Output rows:         "
    f"{total_output_rows:,}"
)

print(
    f"Target-valid rows:   "
    f"{total_target_valid:,}"
)

print(
    f"Feature-valid rows:  "
    f"{total_feature_valid:,}"
)

print(
    f"Detected gaps:       "
    f"{total_gap_rows:,}"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nFeature columns: "
    f"{len(FEATURE_COLUMNS)}"
)

print("\n" + "=" * 70)
