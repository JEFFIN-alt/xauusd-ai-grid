from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/raw/dukascopy")

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close"]


# ============================================================
# HELPERS
# ============================================================

def format_number(value):
    return f"{value:,}"


def validate_file(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    result = {
        "file": file_path.name,
        "rows": 0,
        "missing": 0,
        "duplicates": 0,
        "out_of_order": 0,
        "invalid_ohlc": 0,
        "non_positive": 0,
        "large_gaps": 0,
        "flat_candles": 0,
        "first": None,
        "last": None,
        "status": "PASS",
    }

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Could not read file: {e}")
        result["status"] = "FAIL"
        return result

    result["rows"] = len(df)

    print(f"Rows: {format_number(len(df))}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Column check
    # --------------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        result["status"] = "FAIL"
        return result

    print("Required columns: ✅")

    # --------------------------------------------------------
    # Timestamp conversion
    # --------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    invalid_timestamps = df["datetime"].isna().sum()

    if invalid_timestamps > 0:
        print(f"❌ Invalid timestamps: {invalid_timestamps}")
        result["status"] = "FAIL"
    else:
        print("Timestamps: ✅")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = df[REQUIRED_COLUMNS].isna().sum().sum()
    result["missing"] = int(missing_values)

    if missing_values > 0:
        print(f"❌ Missing values: {missing_values}")
        result["status"] = "FAIL"
    else:
        print("Missing values: 0 ✅")

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    if invalid_timestamps == 0 and len(df) > 0:
        result["first"] = df["datetime"].iloc[0]
        result["last"] = df["datetime"].iloc[-1]

        print(f"First timestamp: {result['first']}")
        print(f"Last timestamp:  {result['last']}")

    # --------------------------------------------------------
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicates = df["timestamp"].duplicated().sum()
    result["duplicates"] = int(duplicates)

    if duplicates > 0:
        print(f"❌ Duplicate timestamps: {duplicates}")
        result["status"] = "FAIL"
    else:
        print("Duplicate timestamps: 0 ✅")

    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    if len(df) > 1:
        out_of_order = (df["timestamp"].diff().dropna() <= 0).sum()
    else:
        out_of_order = 0

    result["out_of_order"] = int(out_of_order)

    if out_of_order > 0:
        print(f"❌ Out-of-order timestamps: {out_of_order}")
        result["status"] = "FAIL"
    else:
        print("Chronological order: ✅")

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_invalid = df[
        ["open", "high", "low", "close"]
    ].isna().sum().sum()

    if numeric_invalid > 0:
        print(f"❌ Invalid numeric OHLC values: {numeric_invalid}")
        result["status"] = "FAIL"
    else:
        print("OHLC numeric values: ✅")

    # --------------------------------------------------------
    # Positive price check
    # --------------------------------------------------------

    non_positive = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    ).sum()

    result["non_positive"] = int(non_positive)

    if non_positive > 0:
        print(f"❌ Non-positive prices: {non_positive}")
        result["status"] = "FAIL"
    else:
        print("Positive prices: ✅")

    # --------------------------------------------------------
    # OHLC logical consistency
    #
    # High must be >= Open and Close
    # Low must be <= Open and Close
    # High must be >= Low
    # --------------------------------------------------------

    invalid_ohlc = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["high"] < df["low"])
    ).sum()

    result["invalid_ohlc"] = int(invalid_ohlc)

    if invalid_ohlc > 0:
        print(f"❌ Invalid OHLC relationships: {invalid_ohlc}")
        result["status"] = "FAIL"
    else:
        print("OHLC relationships: ✅")

    # --------------------------------------------------------
    # Time gaps
    #
    # We report them but DO NOT automatically treat them
    # as errors because XAUUSD has market/session closures.
    # --------------------------------------------------------

    if len(df) > 1:
        gaps = df["datetime"].diff().dropna()

        large_gaps = gaps[gaps > pd.Timedelta(minutes=1)]

        result["large_gaps"] = len(large_gaps)

        print(f"Gaps > 1 minute: {len(large_gaps)}")

        if len(large_gaps) > 0:
            largest_gap = large_gaps.max()

            print(f"Largest gap: {largest_gap}")

            print("Largest gaps:")

            for idx in large_gaps.nlargest(5).index:
                previous_time = df.loc[idx - 1, "datetime"]
                current_time = df.loc[idx, "datetime"]

                gap = current_time - previous_time

                print(
                    f"  {previous_time} → "
                    f"{current_time} = {gap}"
                )

    # --------------------------------------------------------
    # Flat candles
    # --------------------------------------------------------

    flat = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    ).sum()

    result["flat_candles"] = int(flat)

    print(
        f"Completely flat candles: "
        f"{format_number(flat)}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if result["status"] == "PASS":
        print("\nSTATUS: 🟢 PASS")
    else:
        print("\nSTATUS: 🔴 FAIL")

    return result


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("XAUUSD M1 DATASET VALIDATION")
    print("=" * 80)

    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        print(f"\n❌ No CSV files found in {DATA_DIR}")
        return

    print(f"\nFiles found: {len(files)}")

    for file in files:
        print(f"  {file.name}")

    results = []

    # --------------------------------------------------------
    # Validate each file
    # --------------------------------------------------------

    for file_path in files:
        result = validate_file(file_path)
        results.append(result)

    # --------------------------------------------------------
    # Cross-file validation
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CROSS-FILE VALIDATION")
    print("=" * 80)

    ranges = []

    for result in results:

        if result["first"] is not None:

            ranges.append(
                (
                    result["file"],
                    result["first"],
                    result["last"],
                    result["rows"],
                )
            )

    print("\nFile ranges:\n")

    for file_name, first, last, rows in ranges:

        print(
            f"{file_name}\n"
            f"  {first} → {last}\n"
            f"  Rows: {format_number(rows)}\n"
        )

    # --------------------------------------------------------
    # Check overlapping ranges
    # --------------------------------------------------------

    print("Checking for overlapping files...")

    overlap_found = False

    for i in range(len(ranges)):

        file_a, start_a, end_a, rows_a = ranges[i]

        for j in range(i + 1, len(ranges)):

            file_b, start_b, end_b, rows_b = ranges[j]

            if start_a <= end_b and start_b <= end_a:

                print(
                    f"⚠️ OVERLAP:\n"
                    f"  {file_a}\n"
                    f"  {file_b}"
                )

                overlap_found = True

    if not overlap_found:
        print("No overlapping date ranges: ✅")

    # --------------------------------------------------------
    # Check gaps between files
    # --------------------------------------------------------

    print("\nChecking continuity between yearly files...")

    sorted_ranges = sorted(
        ranges,
        key=lambda x: x[1]
    )

    for i in range(len(sorted_ranges) - 1):

        current = sorted_ranges[i]
        next_file = sorted_ranges[i + 1]

        current_end = current[2]
        next_start = next_file[1]

        gap = next_start - current_end

        print(
            f"{current[0]} → {next_file[0]}"
        )
        print(
            f"  End:   {current_end}"
        )
        print(
            f"  Start: {next_start}"
        )
        print(
            f"  Gap:   {gap}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    total_rows = sum(
        result["rows"]
        for result in results
    )

    failed_files = [
        result["file"]
        for result in results
        if result["status"] != "PASS"
    ]

    print(
        f"\nFiles validated: "
        f"{len(results)}"
    )

    print(
        f"Total rows: "
        f"{format_number(total_rows)}"
    )

    print(
        f"Total missing values: "
        f"{sum(r['missing'] for r in results)}"
    )

    print(
        f"Total duplicate timestamps: "
        f"{sum(r['duplicates'] for r in results)}"
    )

    print(
        f"Total invalid OHLC rows: "
        f"{sum(r['invalid_ohlc'] for r in results)}"
    )

    print(
        f"Total non-positive prices: "
        f"{sum(r['non_positive'] for r in results)}"
    )

    print(
        f"Total gaps > 1 minute: "
        f"{sum(r['large_gaps'] for r in results)}"
    )

    print(
        f"Total flat candles: "
        f"{format_number(sum(r['flat_candles'] for r in results))}"
    )

    print()

    if failed_files:
        print("🔴 DATASET VALIDATION: FAIL")

        print("\nFailed files:")

        for file_name in failed_files:
            print(f"  - {file_name}")

    else:
        print("🟢 DATASET VALIDATION: PASS")

    print("\n")


if __name__ == "__main__":
    main()
