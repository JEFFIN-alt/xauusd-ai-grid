#!/usr/bin/env python3
"""
Year-by-year class-distribution diagnostic for triple-barrier targets.

Checks:
- 5 bps control: 30m, 60m
- 10 bps: 30m, 60m
- 15 bps: 30m, 60m
- 20 bps: 30m, 60m

Labels:
- -2 = INVALID
- -1 = AMBIGUOUS
-  0 = SHORT
-  1 = TIMEOUT
-  2 = LONG

Only valid rows (0/1/2) are used for class percentages.
2026 is reported for diagnostics only; it is NOT used for model selection.
"""

from pathlib import Path
import pandas as pd

BASE = Path("data/processed")
OUT = BASE / "triple_barrier_yearly_distribution.csv"

CONFIG = {
    5: BASE / "xauusd_m1_triple_barrier_targets.csv",
    10: BASE / "xauusd_m1_triple_barrier_10bps.csv",
    15: BASE / "xauusd_m1_triple_barrier_15bps.csv",
    20: BASE / "xauusd_m1_triple_barrier_20bps.csv",
}

HORIZONS = [30, 60]
CHUNK_SIZE = 250_000


def label_col(columns, horizon):
    candidates = [
        f"tb_label_{horizon}m",
        f"label_{horizon}m",
        f"tb_label_{horizon}",
    ]
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"Could not find label column for {horizon}m. "
        f"Available columns: {list(columns)}"
    )


def process_file(path: Path, barrier_bps: int):
    print("=" * 78)
    print(f"BARRIER +/- {barrier_bps} BPS")
    print(f"File: {path}")
    print("=" * 78)

    if not path.exists():
        raise FileNotFoundError(path)

    first = pd.read_csv(path, nrows=5)
    if "timestamp" not in first.columns:
        raise KeyError(f"'timestamp' column missing in {path}")

    label_cols = {h: label_col(first.columns, h) for h in HORIZONS}
    print(f"Detected labels: {label_cols}")

    usecols = ["timestamp"] + list(dict.fromkeys(label_cols.values()))

    # key = (year, horizon)
    counts = {}
    total_rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE)
    ):
        total_rows += len(chunk)
        ts = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(f"Invalid timestamps found in {path}")

        chunk["year"] = ts.dt.year.astype("int16")

        for h in HORIZONS:
            col = label_cols[h]
            s = pd.to_numeric(chunk[col], errors="coerce")

            # Initialize per-year integer counts.
            grouped = (
                pd.DataFrame(
                    {
                        "year": chunk["year"],
                        "label": s,
                    }
                )
                .groupby(["year", "label"])
                .size()
            )

            for (year, label), n in grouped.items():
                if pd.isna(label):
                    continue
                key = (int(year), h)
                if key not in counts:
                    counts[key] = {"invalid": 0, "ambiguous": 0,
                                    "short": 0, "timeout": 0, "long": 0}
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
                        f"Unexpected label {label} in {path}, horizon {h}, year {year}"
                    )

        if (chunk_no + 1) % 5 == 0:
            print(f"Chunks processed: {chunk_no + 1:,} | rows: {total_rows:,}")

    rows = []
    for (year, h), c in sorted(counts.items()):
        valid = c["short"] + c["timeout"] + c["long"]
        pct_short = c["short"] / valid * 100 if valid else float("nan")
        pct_timeout = c["timeout"] / valid * 100 if valid else float("nan")
        pct_long = c["long"] / valid * 100 if valid else float("nan")

        rows.append(
            {
                "barrier_bps": barrier_bps,
                "year": year,
                "horizon_min": h,
                "total_rows": (
                    valid + c["invalid"] + c["ambiguous"]
                ),
                "valid_rows": valid,
                "ambiguous_rows": c["ambiguous"],
                "invalid_rows": c["invalid"],
                "short_rows": c["short"],
                "timeout_rows": c["timeout"],
                "long_rows": c["long"],
                "short_pct_valid": pct_short,
                "timeout_pct_valid": pct_timeout,
                "long_pct_valid": pct_long,
            }
        )

    return rows


def print_summary(df: pd.DataFrame):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    for h in HORIZONS:
        print("\n" + "=" * 78)
        print(f"HORIZON {h}m — VALID CLASS DISTRIBUTION BY YEAR")
        print("=" * 78)
        sub = df[df["horizon_min"] == h].copy()
        print(
            sub[
                [
                    "barrier_bps", "year",
                    "short_pct_valid", "timeout_pct_valid", "long_pct_valid",
                    "valid_rows", "ambiguous_rows", "invalid_rows",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )


def main():
    print("=" * 78)
    print("XAUUSD TRIPLE-BARRIER YEAR-BY-YEAR TARGET DIAGNOSTICS")
    print("=" * 78)
    print("Barriers: +/- 5, 10, 15, 20 bps")
    print("Horizons: 30, 60 minutes")
    print("Valid labels: SHORT / TIMEOUT / LONG")
    print("2026: DIAGNOSTIC ONLY — NOT USED FOR MODEL SELECTION")
    print("=" * 78)

    all_rows = []

    for barrier_bps, path in CONFIG.items():
        all_rows.extend(process_file(path, barrier_bps))

    df = pd.DataFrame(all_rows).sort_values(
        ["horizon_min", "barrier_bps", "year"]
    ).reset_index(drop=True)

    # Sanity checks.
    pct_sum = (
        df["short_pct_valid"]
        + df["timeout_pct_valid"]
        + df["long_pct_valid"]
    )
    if not ((pct_sum - 100).abs() < 1e-6).all():
        raise AssertionError("Class percentages do not sum to 100% for all rows.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print_summary(df)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print(f"Saved: {OUT}")
    print("2026 is included only for inspection and remains excluded from model selection.")
    print("=" * 78)


if __name__ == "__main__":
    main()
