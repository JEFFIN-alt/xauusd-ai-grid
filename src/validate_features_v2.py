import pandas as pd
import numpy as np


V2_FILE = "data/processed/xauusd_m1_15m_features_v2.csv"

TARGET_FILE = "data/processed/xauusd_m1_15m_target.csv"

CHUNK_SIZE = 100_000


print("=" * 70)
print("VALIDATING XAUUSD FEATURE DATASET V2")
print("=" * 70)


# ============================================================
# BASIC DATASET CHECK
# ============================================================

columns = pd.read_csv(
    V2_FILE,
    nrows=1
).columns.tolist()


print("\nColumns:", len(columns))

print("\nColumn list:")

for column in columns:
    print(" ", column)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",

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

    "future_return_15m",
]


missing_columns = [
    c for c in required
    if c not in columns
]


if missing_columns:

    print("\nERROR: Missing columns:")

    for column in missing_columns:
        print(" ", column)

    raise SystemExit(1)


# ============================================================
# STREAM VALIDATION
# ============================================================

feature_columns = [
    c for c in required
    if c not in [
        "timestamp",
        "future_return_15m",
    ]
]


total_rows = 0

missing_values = 0

infinite_values = 0

duplicate_timestamps = 0

chronological = True

first_timestamp = None

last_timestamp = None

previous_timestamp = None


# ============================================================
# VALIDATE V2
# ============================================================

print("\nChecking V2 dataset...")


for chunk_number, chunk in enumerate(

    pd.read_csv(
        V2_FILE,
        usecols=required,
        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    total_rows += len(chunk)


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values += int(
        chunk.isna().sum().sum()
    )


    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_columns = [
        c for c in required
        if c != "timestamp"
    ]

    numeric_data = chunk[
        numeric_columns
    ]

    infinite_values += int(
        np.isinf(
            numeric_data.to_numpy()
        ).sum()
    )


    # --------------------------------------------------------
    # Timestamp checks
    # --------------------------------------------------------

    timestamps = chunk[
        "timestamp"
    ].to_numpy()


    if len(timestamps) > 0:

        if first_timestamp is None:
            first_timestamp = timestamps[0]

        if (
            previous_timestamp is not None
            and timestamps[0] <= previous_timestamp
        ):

            chronological = False

        if len(timestamps) > 1:

            if np.any(
                np.diff(timestamps) <= 0
            ):
                chronological = False


        duplicate_timestamps += int(
            chunk["timestamp"]
            .duplicated()
            .sum()
        )


        previous_timestamp = timestamps[-1]

        last_timestamp = timestamps[-1]


    if chunk_number % 10 == 0:

        print(
            f"Processed chunks: "
            f"{chunk_number} | "
            f"Rows: {total_rows:,}"
        )


# ============================================================
# CROSS-CHECK AGAINST ORIGINAL TARGET
# ============================================================

print("\nComparing V2 targets against official target...")


target_usecols = [
    "timestamp",
    "future_return_15m",
]


official_target = {}

# We only need timestamps and target values.
# Store the official target in a dictionary.
# This is larger than ideal, but only has 2 columns.

for chunk in pd.read_csv(
    TARGET_FILE,
    usecols=target_usecols,
    chunksize=CHUNK_SIZE,
):

    for timestamp, value in zip(
        chunk["timestamp"].to_numpy(),
        chunk["future_return_15m"].to_numpy(),
    ):

        official_target[int(timestamp)] = float(value)


print(
    f"Official target rows loaded: "
    f"{len(official_target):,}"
)


comparison_rows = 0

mismatches = 0

missing_from_official = 0

max_difference = 0.0


for chunk in pd.read_csv(
    V2_FILE,
    usecols=[
        "timestamp",
        "future_return_15m",
    ],
    chunksize=CHUNK_SIZE,
):

    for timestamp, value in zip(
        chunk["timestamp"].to_numpy(),
        chunk["future_return_15m"].to_numpy(),
    ):

        timestamp = int(timestamp)

        value = float(value)

        comparison_rows += 1

        if timestamp not in official_target:

            missing_from_official += 1

            continue


        difference = abs(
            value -
            official_target[timestamp]
        )


        if difference > max_difference:

            max_difference = difference


        if difference > 1e-12:

            mismatches += 1


print(
    f"\nV2 rows compared: "
    f"{comparison_rows:,}"
)

print(
    f"Missing from official target: "
    f"{missing_from_official:,}"
)

print(
    f"Target mismatches: "
    f"{mismatches:,}"
)

print(
    f"Maximum target difference: "
    f"{max_difference:.12e}"
)


# ============================================================
# TIMESTAMP RANGE
# ============================================================

first_dt = pd.to_datetime(
    first_timestamp,
    unit="ms",
    utc=True
)

last_dt = pd.to_datetime(
    last_timestamp,
    unit="ms",
    utc=True
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("V2 VALIDATION RESULT")
print("=" * 70)

print(
    f"\nRows:                 "
    f"{total_rows:,}"
)

print(
    f"Columns:              "
    f"{len(columns):,}"
)

print(
    f"First timestamp:      "
    f"{first_dt}"
)

print(
    f"Last timestamp:       "
    f"{last_dt}"
)

print(
    f"Missing values:       "
    f"{missing_values:,}"
)

print(
    f"Infinite values:      "
    f"{infinite_values:,}"
)

print(
    f"Duplicate timestamps: "
    f"{duplicate_timestamps:,}"
)

print(
    f"Chronological:        "
    f"{chronological}"
)

print(
    f"Target mismatches:    "
    f"{mismatches:,}"
)

print(
    f"Missing official:     "
    f"{missing_from_official:,}"
)

print(
    f"Max target difference: "
    f"{max_difference:.12e}"
)


print("\nExpected:")

print("  Missing values       = 0")

print("  Infinite values      = 0")

print("  Duplicate timestamps = 0")

print("  Chronological        = True")

print("  Target mismatches    = 0")

print("  Missing official     = 0")


print("\n" + "=" * 70)
