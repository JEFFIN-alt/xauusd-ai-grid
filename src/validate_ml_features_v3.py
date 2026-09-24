import pandas as pd
import numpy as np


V3_FILE = (
    "data/processed/"
    "xauusd_m1_ml_features_v3.csv"
)

CHUNK_SIZE = 100_000

HORIZONS = [5, 15, 30, 60]


FEATURES = [
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

BASE = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
]

TARGETS = [
    f"future_return_{h}m"
    for h in HORIZONS
]

REQUIRED = BASE + FEATURES + TARGETS


print("=" * 70)
print("VALIDATING XAUUSD ML DATASET V3")
print("=" * 70)


# ============================================================
# COLUMN CHECK
# ============================================================

columns = pd.read_csv(
    V3_FILE,
    nrows=1
).columns.tolist()

print(
    f"\nColumns found: {len(columns)}"
)

print(
    f"Columns expected: {len(REQUIRED)}"
)

missing_columns = [
    c for c in REQUIRED
    if c not in columns
]

if missing_columns:

    print("\nMissing columns:")

    for c in missing_columns:
        print(" ", c)

    raise SystemExit(1)


# ============================================================
# STREAM VALIDATION
# ============================================================

total_rows = 0

base_feature_missing = 0

target_nan_counts = {
    h: 0
    for h in HORIZONS
}

target_valid_counts = {
    h: 0
    for h in HORIZONS
}

infinite_values = 0

duplicate_timestamps = 0

backward_timestamps = 0

non_minute_timestamps = 0

first_timestamp = None
last_timestamp = None
previous_timestamp = None


print("\nScanning V3...")


for chunk_number, chunk in enumerate(
    pd.read_csv(
        V3_FILE,
        usecols=REQUIRED,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    total_rows += len(chunk)


    # --------------------------------------------------------
    # BASE + FEATURE MISSING VALUES
    # --------------------------------------------------------

    base_feature_missing += int(
        chunk[
            BASE + FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )


    # --------------------------------------------------------
    # INFINITE VALUES
    #
    # Targets are allowed to be NaN, but any finite target
    # must not be infinite.
    # --------------------------------------------------------

    numeric_columns = [
        c for c in BASE + FEATURES + TARGETS
        if c != "timestamp"
    ]

    numeric = chunk[
        numeric_columns
    ].to_numpy(
        dtype=np.float64
    )

    infinite_values += int(
        np.isinf(numeric).sum()
    )


    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    ts = chunk[
        "timestamp"
    ].to_numpy(
        dtype=np.int64
    )

    if len(ts):

        if first_timestamp is None:
            first_timestamp = ts[0]

        last_timestamp = ts[-1]


        non_minute_timestamps += int(
            (ts % 60_000 != 0).sum()
        )


        if previous_timestamp is not None:

            boundary_diff = (
                ts[0]
                - previous_timestamp
            )

            if boundary_diff == 0:
                duplicate_timestamps += 1

            elif boundary_diff < 0:
                backward_timestamps += 1


        if len(ts) > 1:

            diffs = np.diff(ts)

            duplicate_timestamps += int(
                (diffs == 0).sum()
            )

            backward_timestamps += int(
                (diffs < 0).sum()
            )


        previous_timestamp = ts[-1]


    # --------------------------------------------------------
    # TARGET COUNTS
    # --------------------------------------------------------

    for horizon in HORIZONS:

        target = (
            f"future_return_{horizon}m"
        )

        values = chunk[
            target
        ].to_numpy(
            dtype=np.float64
        )

        finite = np.isfinite(values)

        target_valid_counts[horizon] += int(
            finite.sum()
        )

        target_nan_counts[horizon] += int(
            np.isnan(values).sum()
        )


    if chunk_number % 10 == 0:

        print(
            f"Processed chunks: "
            f"{chunk_number} | "
            f"Rows: {total_rows:,}"
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("V3 VALIDATION RESULT")
print("=" * 70)

print(
    f"\nRows:                  "
    f"{total_rows:,}"
)

print(
    f"Columns:               "
    f"{len(columns)}"
)

print(
    "First timestamp:       "
    f"{pd.to_datetime(first_timestamp, unit='ms', utc=True)}"
)

print(
    "Last timestamp:        "
    f"{pd.to_datetime(last_timestamp, unit='ms', utc=True)}"
)

print(
    f"Base/feature missing:  "
    f"{base_feature_missing:,}"
)

print(
    f"Infinite values:       "
    f"{infinite_values:,}"
)

print(
    f"Non-minute timestamps: "
    f"{non_minute_timestamps:,}"
)

print(
    f"Duplicate timestamps:  "
    f"{duplicate_timestamps:,}"
)

print(
    f"Backward timestamps:   "
    f"{backward_timestamps:,}"
)


print("\nTarget availability:")

for horizon in HORIZONS:

    print(
        f"  {horizon:>2}m valid: "
        f"{target_valid_counts[horizon]:,}"
    )

    print(
        f"  {horizon:>2}m NaN:   "
        f"{target_nan_counts[horizon]:,}"
    )


print("\nExpected interpretation:")

print("  Base/feature missing  = 0")
print("  Infinite values       = 0")
print("  Non-minute timestamps = 0")
print("  Duplicate timestamps  = 0")
print("  Backward timestamps   = 0")
print("  Target NaNs           = allowed/expected")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
