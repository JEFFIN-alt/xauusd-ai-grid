#!/usr/bin/env python3
"""
Corrected year-by-year class-distribution diagnostic for triple-barrier targets.

Important:
The first version parsed numeric epoch timestamps with pandas' default
nanosecond interpretation, which can turn millisecond/second epochs into 1970.
This version detects numeric epoch units from timestamp magnitude.

Checks:
- 5 bps control
- 10, 15, 20 bps
- horizons 30m and 60m
- years 2019-2026 (2026 diagnostic only)
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")
OUT = BASE / "triple_barrier_yearly_distribution.csv"

CONFIG = {
    5: BASE / "xauusd_m1_triple_barrier_targets.csv",
    10: BASE / "xauusd_m1_triple_barrier_10bps.csv",
    15: BASE / "xauusd_m1_triple_barrier_15bps.csv",
}

HORIZONS = [30, 60]
CHUNK_SIZE = 250_000


def detect_epoch_unit(values):
    """Detect seconds/ms/us/ns for numeric epoch timestamps."""
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return None

    med = float(s.abs().median())
    if med >= 1e17:
        return "ns"
    if med >= 1e14:
        return "us"
    if med >= 1e11:
        return "ms"
    if med >= 1e8:
        return "s"
    return None


def parse_timestamp(values):
    """Robustly parse string or numeric epoch timestamps."""
    numeric = pd.to_numeric(values, errors="coerce")

    # If nearly all values are numeric, treat them as epoch values.
    numeric_ratio = numeric.notna().mean()
    if numeric_ratio > 0.99:
        unit = detect_epoch_unit(numeric)
        if unit is None:
            raise ValueError(
                f"Could not determine epoch unit; median={numeric.abs().median()}"
            )
        ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
        return ts, f"epoch-{unit}"

    # Otherwise parse ISO-like/string timestamps.
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    return ts, "datetime-string"


def label_col(columns, horizon):
    for c in (
        f"tb_label_{horizon}m",
        f"label_{horizon}m",
        f"tb_label_{horizon}",
    ):
        if c in columns:
            return c
    raise KeyError(
        f"Could not find label column for {horizon}m. Available: {list(columns)}"
    )


def process_file(path, barrier_bps):
    print("=" * 78)
    print(f"BARRIER +/- {barrier_bps} BPS")
    print(f"File: {path}")
    print("=" * 78)

    if not path.exists():
        raise FileNotFoundError(path)

    first = pd.read_csv(path, nrows=5)
    label_cols = {h: label_col(first.columns, h) for h in HORIZONS}
    print(f"Detected labels: {label_cols}")

    usecols = ["timestamp"] + list(label_cols.values())
    counts = {}
    total_rows = 0
    detected_timestamp_mode = None

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE)
    ):
        total_rows += len(chunk)

        ts, timestamp_mode = parse_timestamp(chunk["timestamp"])
        detected_timestamp_mode = timestamp_mode
        if ts.isna().any():
            bad = int(ts.isna().sum())
            raise ValueError(
                f"{bad} invalid timestamps in {path}; parse mode={timestamp_mode}"
            )

        years = ts.dt.year.astype("int16")

        # Sanity guard: target data should fall in the expected project period.
        if not years.between(2019, 2026).all():
            bad_years = sorted(years.unique().tolist())[:20]
            raise ValueError(
                f"Unexpected years {bad_years} in {path}. "
                f"Detected timestamp mode={timestamp_mode}"
            )

        for h in HORIZONS:
            labels = pd.to_numeric(chunk[label_cols[h]], errors="coerce")
            tmp = pd.DataFrame({"year": years, "label": labels})

            grouped = tmp.groupby(["year", "label"]).size()
            for (year, label), n in grouped.items():
                if pd.isna(label):
                    continue
                key = (int(year), h)
                if key not in counts:
                    counts[key] = {
                        "invalid": 0, "ambiguous": 0,
                        "short": 0, "timeout": 0, "long": 0
                    }

                n = int(n)
                label = int(label)
                if label == -2:
                    counts[key]["invalid"] += n
                elif label == -1:
                    counts[key]["ambiguous"] += n
                elif label == 0:
                    counts[key]["short"] += n
                elif label == 1:
                    counts[key]["timeout"] += n
                elif label == 2:
                    counts[key]["long"] += n
                else:
                    raise ValueError(
                        f"Unexpected label {label} in {path}, horizon={h}, year={year}"
                    )

        if (chunk_no + 1) % 5 == 0:
            print(f"Chunks processed: {chunk_no + 1:,} | rows: {total_rows:,}")

    print(f"Timestamp parse mode: {detected_timestamp_mode}")

    rows = []
    for (year, h), c in sorted(counts.items()):
        valid = c["short"] + c["timeout"] + c["long"]
        if valid == 0:
            continue

        rows.append({
            "barrier_bps": barrier_bps,
            "year": year,
            "horizon_min": h,
            "valid_rows": valid,
            "ambiguous_rows": c["ambiguous"],
            "invalid_rows": c["invalid"],
            "short_rows": c["short"],
            "timeout_rows": c["timeout"],
            "long_rows": c["long"],
            "short_pct_valid": c["short"] / valid * 100.0,
            "timeout_pct_valid": c["timeout"] / valid * 100.0,
            "long_pct_valid": c["long"] / valid * 100.0,
        })

    return rows


def main():
    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER YEARLY TARGET DIAGNOSTICS V2")
    print("=" * 78)
    print("Barriers: +/- 5, 10, 15 bps")
    print("Horizons: 30, 60 minutes")
    print("Timestamp parser: auto-detect epoch unit / ISO datetime")
    print("2026: diagnostic only")
    print("=" * 78)

    all_rows = []
    for barrier_bps, path in CONFIG.items():
        all_rows.extend(process_file(path, barrier_bps))

    df = pd.DataFrame(all_rows).sort_values(
        ["horizon_min", "barrier_bps", "year"]
    ).reset_index(drop=True)

    pct_sum = (
        df["short_pct_valid"]
        + df["timeout_pct_valid"]
        + df["long_pct_valid"]
    )
    if not np.allclose(pct_sum.to_numpy(), 100.0, atol=1e-6):
        raise AssertionError("Class percentages do not sum to 100%.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    for h in HORIZONS:
        print("\n" + "=" * 78)
        print(f"HORIZON {h}m — VALID CLASS DISTRIBUTION BY YEAR")
        print("=" * 78)
        sub = df[df["horizon_min"] == h]
        print(
            sub[
                [
                    "barrier_bps", "year",
                    "short_pct_valid", "timeout_pct_valid", "long_pct_valid",
                    "valid_rows", "ambiguous_rows", "invalid_rows",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

    print("\n" + "=" * 78)
    print("DIAGNOSTIC V2 COMPLETE")
    print("=" * 78)
    print(f"Saved: {OUT}")
    print("2026 is included only for inspection and remains excluded from model selection.")
    print("=" * 78)


if __name__ == "__main__":
    main()
