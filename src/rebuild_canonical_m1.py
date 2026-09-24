from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

RAW_DIR = Path("data/raw/dukascopy")

OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = (
    OUTPUT_DIR /
    "xauusd_m1_2019_2026_canonical.csv"
)

CHUNK_SIZE = 100_000


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()


files = sorted(
    RAW_DIR.glob("*.csv")
)


if not files:
    raise FileNotFoundError(
        f"No raw CSV files found in {RAW_DIR}"
    )


print("=" * 70)
print("BUILDING CANONICAL XAUUSD M1 DATASET")
print("=" * 70)

print(
    f"\nRaw files: {len(files)}"
)

for file in files:
    print(
        f"  {file.name}"
    )


# ============================================================
# STATE
# ============================================================

first_output = True

previous_timestamp = None

total_rows = 0

written_rows = 0

invalid_timestamp_rows = 0

duplicate_rows = 0

backward_rows = 0

invalid_ohlc_rows = 0

non_positive_rows = 0

missing_rows = 0

flat_rows = 0

gap_rows = 0

max_gap_ms = 0


# ============================================================
# PROCESS EACH RAW FILE
# ============================================================

for file_number, file in enumerate(
    files,
    start=1
):

    print(
        f"\n[{file_number}/{len(files)}] "
        f"Processing {file.name}"
    )

    file_previous_timestamp = None


    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file,

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

        total_rows += len(chunk)


        # ----------------------------------------------------
        # NUMERIC CONVERSION
        # ----------------------------------------------------

        for column in [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]:

            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )


        # ----------------------------------------------------
        # MISSING
        # ----------------------------------------------------

        row_missing = (
            chunk[
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ]
            .isna()
            .any(axis=1)
        )

        missing_rows += int(
            row_missing.sum()
        )


        # ----------------------------------------------------
        # TIMESTAMP VALIDITY
        # ----------------------------------------------------

        timestamp = chunk[
            "timestamp"
        ]

        valid_numeric_timestamp = (
            timestamp.notna()
        )

        minute_aligned = (
            timestamp.mod(60_000) == 0
        )

        valid_timestamp = (
            valid_numeric_timestamp
            &
            minute_aligned
        )

        invalid_timestamp_rows += int(
            (~valid_timestamp).sum()
        )


        # ----------------------------------------------------
        # CHECK TIMESTAMP ORDER
        # ----------------------------------------------------

        valid_ts_values = (
            timestamp[
                valid_timestamp
            ]
            .astype("int64")
            .to_numpy()
        )


        if len(valid_ts_values) > 1:

            diffs = np.diff(
                valid_ts_values
            )

            duplicate_rows += int(
                (diffs == 0).sum()
            )

            backward_rows += int(
                (diffs < 0).sum()
            )

            positive_diffs = (
                diffs[diffs > 0]
            )

            if len(positive_diffs):

                gap_rows += int(
                    (
                        positive_diffs
                        > 60_000
                    ).sum()
                )

                local_max_gap = int(
                    positive_diffs.max()
                )

                max_gap_ms = max(
                    max_gap_ms,
                    local_max_gap
                )


        # ----------------------------------------------------
        # CROSS-CHUNK / CROSS-FILE TIMESTAMP ORDER
        # ----------------------------------------------------

        if (
            previous_timestamp
            is not None
            and
            len(valid_ts_values) > 0
        ):

            first_valid = int(
                valid_ts_values[0]
            )

            diff = (
                first_valid -
                previous_timestamp
            )

            if diff == 0:

                duplicate_rows += 1

            elif diff < 0:

                backward_rows += 1

            elif diff > 60_000:

                gap_rows += 1

                max_gap_ms = max(
                    max_gap_ms,
                    diff
                )


        if len(valid_ts_values) > 0:

            previous_timestamp = int(
                valid_ts_values[-1]
            )


        # ----------------------------------------------------
        # OHLC VALIDITY
        # ----------------------------------------------------

        valid_ohlc = (
            (chunk["high"] >= chunk["open"])
            &
            (chunk["high"] >= chunk["close"])
            &
            (chunk["low"] <= chunk["open"])
            &
            (chunk["low"] <= chunk["close"])
            &
            (chunk["high"] >= chunk["low"])
        )

        invalid_ohlc_rows += int(
            (~valid_ohlc).sum()
        )


        # ----------------------------------------------------
        # POSITIVE PRICE
        # ----------------------------------------------------

        positive_price = (
            (chunk["open"] > 0)
            &
            (chunk["high"] > 0)
            &
            (chunk["low"] > 0)
            &
            (chunk["close"] > 0)
        )

        non_positive_rows += int(
            (~positive_price).sum()
        )


        # ----------------------------------------------------
        # FLAT CANDLES
        # ----------------------------------------------------

        flat = (
            (chunk["open"] == chunk["high"])
            &
            (chunk["high"] == chunk["low"])
            &
            (chunk["low"] == chunk["close"])
        )

        flat_rows += int(
            flat.sum()
        )


        # ----------------------------------------------------
        # ONLY WRITE VALID SOURCE RECORDS
        # ----------------------------------------------------

        write_mask = (
            valid_timestamp
            &
            valid_ohlc
            &
            positive_price
        )


        output = chunk.loc[
            write_mask,
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        ].copy()


        if not output.empty:

            output.to_csv(
                OUTPUT_FILE,

                mode=(
                    "w"
                    if first_output
                    else "a"
                ),

                header=first_output,

                index=False,

            )

            first_output = False

            written_rows += len(
                output
            )


        if chunk_number % 5 == 0:

            print(
                f"  Chunks: {chunk_number} | "
                f"Input: {total_rows:,} | "
                f"Written: {written_rows:,}"
            )


    # --------------------------------------------------------
    # FILE-LEVEL ORDER CHECK
    # --------------------------------------------------------

    print(
        f"  Completed: {file.name}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CANONICAL DATASET BUILD COMPLETE")
print("=" * 70)

print(
    f"\nInput rows: "
    f"{total_rows:,}"
)

print(
    f"Written rows: "
    f"{written_rows:,}"
)

print(
    f"Invalid timestamp rows: "
    f"{invalid_timestamp_rows:,}"
)

print(
    f"Duplicate timestamp transitions: "
    f"{duplicate_rows:,}"
)

print(
    f"Backward timestamp transitions: "
    f"{backward_rows:,}"
)

print(
    f"Invalid OHLC rows: "
    f"{invalid_ohlc_rows:,}"
)

print(
    f"Non-positive price rows: "
    f"{non_positive_rows:,}"
)

print(
    f"Flat candles: "
    f"{flat_rows:,}"
)

print(
    f"Gaps > 1 minute: "
    f"{gap_rows:,}"
)

print(
    f"Maximum gap: "
    f"{max_gap_ms / 60_000:.2f} minutes"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
