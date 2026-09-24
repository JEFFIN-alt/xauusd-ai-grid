import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "data/processed/xauusd_m1_2019_2026_canonical.csv"
)

OUTPUT_FILE = Path(
    "data/processed/xauusd_m1_ml_features_v3.csv"
)

CHUNK_SIZE = 100_000

HORIZONS = [5, 15, 30, 60]

MAX_LOOKBACK = 480

MAX_HORIZON = max(HORIZONS)


# ============================================================
# FEATURES
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


TARGET_COLUMNS = [
    f"future_return_{h}m"
    for h in HORIZONS
]


# ============================================================
# HELPERS
# ============================================================

def grouped_shift(series, groups, periods):

    return (
        series
        .groupby(groups, sort=False)
        .shift(periods)
    )


def grouped_rolling_std(series, groups, window):

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
print("BUILDING FINAL XAUUSD ML DATASET V3")
print("=" * 70)

print(f"\nInput:")
print(INPUT_FILE)

print(f"\nOutput:")
print(OUTPUT_FILE)

print(
    f"\nHorizons: {HORIZONS}"
)

print(
    f"Maximum feature lookback: "
    f"{MAX_LOOKBACK} consecutive M1 bars"
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_FILE.exists():

    OUTPUT_FILE.unlink()

    print(
        "\nRemoved existing V3 output."
    )


# ============================================================
# STATE
# ============================================================

carry = pd.DataFrame()

first_output = True

last_written_timestamp = None

total_input_rows = 0

total_output_rows = 0

feature_valid_rows = 0

target_valid_counts = {
    h: 0
    for h in HORIZONS
}


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
    # Combine historical carry + current chunk
    # --------------------------------------------------------

    if carry.empty:

        working = chunk.copy()

    else:

        working = pd.concat(
            [
                carry,
                chunk,
            ],
            ignore_index=True
        )


    # --------------------------------------------------------
    # Chronological safety
    # --------------------------------------------------------

    timestamps = (
        working["timestamp"]
        .to_numpy(
            dtype=np.int64
        )
    )

    if len(timestamps) > 1:

        diffs = np.diff(
            timestamps
        )

        if np.any(diffs <= 0):

            raise RuntimeError(
                "Canonical dataset is not strictly chronological."
            )


    # --------------------------------------------------------
    # CONTINUOUS SEGMENTS
    #
    # New segment begins whenever the next observation
    # is not exactly one minute later.
    # --------------------------------------------------------

    timestamp_diff = (
        working["timestamp"].diff()
    )

    new_segment = (
        timestamp_diff != 60_000
    )

    new_segment.iloc[0] = True

    working["segment_id"] = (
        new_segment
        .cumsum()
        .astype("int64")
    )

    segment = working["segment_id"]

    close = working["close"]

    high = working["high"]

    low = working["low"]

    open_price = working["open"]


    # ========================================================
    # BACKWARD RETURN FEATURES
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
            - 1.0
        )


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

    body_high = pd.concat(
        [
            open_price,
            close,
        ],
        axis=1
    ).max(axis=1)

    body_low = pd.concat(
        [
            open_price,
            close,
        ],
        axis=1
    ).min(axis=1)

    working["upper_wick"] = (
        high - body_high
    )

    working["lower_wick"] = (
        body_low - low
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
    # DISTANCE FROM SMA
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
    # SMA SLOPE
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
    # DISTANCES FROM HIGH / LOW
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
    # MULTI-HORIZON TARGETS
    # ========================================================

    for horizon in HORIZONS:

        future_close = (
            close.shift(-horizon)
        )

        future_timestamp = (
            working["timestamp"]
            .shift(-horizon)
        )

        future_segment = (
            segment.shift(-horizon)
        )

        expected_diff = (
            horizon *
            60 *
            1000
        )

        actual_diff = (
            future_timestamp -
            working["timestamp"]
        )

        valid_target = (
            (actual_diff == expected_diff)
            &
            (future_segment == segment)
        )

        working[
            f"future_return_{horizon}m"
        ] = np.where(

            valid_target,

            (
                future_close /
                close
            ) - 1.0,

            np.nan,
        )


    # ========================================================
    # FINALIZED ROWS
    #
    # We retain enough rows at the end of the chunk so
    # future targets can be resolved by the next chunk.
    # ========================================================

    finalized_count = (
        len(working)
        - MAX_HORIZON
    )


    if finalized_count > 0:

        finalized_mask = (
            np.arange(len(working))
            < finalized_count
        )

    else:

        finalized_mask = np.zeros(
            len(working),
            dtype=bool
        )


    # --------------------------------------------------------
    # Feature validity
    # --------------------------------------------------------

    feature_valid = (
        working[
            FEATURE_COLUMNS
        ]
        .notna()
        .all(axis=1)
    )


    # --------------------------------------------------------
    # At least one valid target
    # --------------------------------------------------------

    target_valid_any = (
        working[
            TARGET_COLUMNS
        ]
        .notna()
        .any(axis=1)
    )


    final_valid = (
        finalized_mask
        &
        feature_valid
        &
        target_valid_any
    )


    feature_valid_rows += int(
        (
            finalized_mask &
            feature_valid
        ).sum()
    )


    # ========================================================
    # PREPARE OUTPUT
    # ========================================================

    if final_valid.any():

        output_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]

        output_columns += (
            FEATURE_COLUMNS
        )

        output_columns += (
            TARGET_COLUMNS
        )

        current = working.loc[
            final_valid,
            output_columns
        ].copy()


        # Count targets
        for horizon in HORIZONS:

            target = (
                f"future_return_{horizon}m"
            )

            target_valid_counts[horizon] += int(
                current[target]
                .notna()
                .sum()
            )


        # Prevent duplicate timestamps
        if last_written_timestamp is not None:

            current = current.loc[
                current["timestamp"]
                > last_written_timestamp
            ]


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

                os.fsync(
                    handle.fileno()
                )

            first_output = False

            total_output_rows += len(
                current
            )

            last_written_timestamp = int(
                current["timestamp"].iloc[-1]
            )


    # ========================================================
    # CARRY
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


    print(
        f"Chunk {chunk_number:02d} | "
        f"Input: {total_input_rows:,} | "
        f"Output: {total_output_rows:,}"
    )


# ============================================================
# FINAL FLUSH
#
# Last 60 rows can still have shorter-horizon targets.
# ============================================================

if not carry.empty:

    working = carry.copy()

    timestamp = working["timestamp"]

    close = working["close"]

    segment_diff = timestamp.diff()

    new_segment = (
        segment_diff != 60_000
    )

    new_segment.iloc[0] = True

    segment = (
        new_segment
        .cumsum()
        .astype("int64")
    )


    # Calculate targets only.
    for horizon in HORIZONS:

        future_close = (
            close.shift(-horizon)
        )

        future_timestamp = (
            timestamp.shift(-horizon)
        )

        future_segment = (
            segment.shift(-horizon)
        )

        expected_diff = (
            horizon * 60 * 1000
        )

        valid = (
            (future_timestamp - timestamp
             == expected_diff)
            &
            (future_segment == segment)
        )

        working[
            f"future_return_{horizon}m"
        ] = np.where(

            valid,

            future_close / close - 1.0,

            np.nan,
        )


    # The feature rows themselves were already computed
    # in the previous chunk; reconstruct them from the
    # previously written feature rows is unnecessary.
    #
    # Therefore final flush is intentionally NOT used
    # for feature dataset construction.
    #
    # The preceding chunks already produced every row
    # that has complete 480-bar feature history and at
    # least one resolved target.


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL ML DATASET V3 COMPLETE")
print("=" * 70)

print(
    f"\nInput rows: "
    f"{total_input_rows:,}"
)

print(
    f"Output rows: "
    f"{total_output_rows:,}"
)

print(
    f"Feature-valid finalized rows: "
    f"{feature_valid_rows:,}"
)

print("\nValid targets in output:")

for horizon in HORIZONS:

    print(
        f"  {horizon:>2}m: "
        f"{target_valid_counts[horizon]:,}"
    )


print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nFeature columns: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Target columns: "
    f"{len(TARGET_COLUMNS)}"
)

print("\n" + "=" * 70)
