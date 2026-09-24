#!/usr/bin/env python3
"""
Year-by-year diagnostic for SESSION-AWARE dynamic triple-barrier targets.

Checks the five 60-minute dynamic configurations:
  0.50x, 0.75x, 1.00x, 1.25x, 1.50x

Valid labels:
  0 SHORT
  1 TIMEOUT
  2 LONG

Excluded:
  -3 SESSION_CLOSED
  -2 INVALID
  -1 AMBIGUOUS

2026 is displayed for diagnostics only and must not be used for later
model selection/testing.

Outputs:
  data/processed/xauusd_m1_session_dynamic_tb_yearly_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("data/processed")

FILES = {
    0.50: BASE / "xauusd_m1_session_dynamic_tb_0p50.csv",
    0.75: BASE / "xauusd_m1_session_dynamic_tb_0p75.csv",
    1.00: BASE / "xauusd_m1_session_dynamic_tb_1p00.csv",
    1.25: BASE / "xauusd_m1_session_dynamic_tb_1p25.csv",
    1.50: BASE / "xauusd_m1_session_dynamic_tb_1p50.csv",
}

LABEL_COL = "tb_label_60m"
CHUNK_SIZE = 250_000

OUT = BASE / "xauusd_m1_session_dynamic_tb_yearly_summary.csv"


def parse_epoch_ms(values):
    x = pd.to_numeric(values, errors="coerce")
    if x.isna().any():
        raise ValueError("Invalid numeric timestamp found.")
    return x.astype(np.int64)


def process_file(path, multiplier):
    if not path.exists():
        raise FileNotFoundError(path)

    head = pd.read_csv(path, nrows=2)
    required = ["timestamp", LABEL_COL]
    missing = [c for c in required if c not in head.columns]
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")

    usecols = required

    counts = {}
    total_rows = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE)
    ):
        ts = parse_epoch_ms(chunk["timestamp"])
        labels = pd.to_numeric(chunk[LABEL_COL], errors="coerce").fillna(-99).astype(np.int16)

        years = pd.to_datetime(ts, unit="ms", utc=True).dt.year.to_numpy(np.int16)

        total_rows += len(chunk)

        for year in np.unique(years):
            m = years == year
            key = int(year)
            if key not in counts:
                counts[key] = {
                    "total_rows": 0,
                    "session_closed_rows": 0,
                    "invalid_rows": 0,
                    "ambiguous_rows": 0,
                    "short_rows": 0,
                    "timeout_rows": 0,
                    "long_rows": 0,
                }

            y = labels[m]
            counts[key]["total_rows"] += int(m.sum())
            counts[key]["session_closed_rows"] += int((y == -3).sum())
            counts[key]["invalid_rows"] += int((y == -2).sum())
            counts[key]["ambiguous_rows"] += int((y == -1).sum())
            counts[key]["short_rows"] += int((y == 0).sum())
            counts[key]["timeout_rows"] += int((y == 1).sum())
            counts[key]["long_rows"] += int((y == 2).sum())

        if (chunk_no + 1) % 5 == 0:
            print(
                f"{multiplier:.2f}x | chunks={chunk_no+1} | "
                f"rows={total_rows:,}"
            )

    rows = []

    for year in sorted(counts):
        c = counts[year]
        valid = c["short_rows"] + c["timeout_rows"] + c["long_rows"]

        rows.append(
            {
                "multiplier": multiplier,
                "year": year,
                "total_rows": c["total_rows"],
                "session_closed_rows": c["session_closed_rows"],
                "invalid_rows": c["invalid_rows"],
                "ambiguous_rows": c["ambiguous_rows"],
                "valid_rows": valid,
                "short_rows": c["short_rows"],
                "timeout_rows": c["timeout_rows"],
                "long_rows": c["long_rows"],
                "short_pct_valid": c["short_rows"] / valid * 100.0 if valid else np.nan,
                "timeout_pct_valid": c["timeout_rows"] / valid * 100.0 if valid else np.nan,
                "long_pct_valid": c["long_rows"] / valid * 100.0 if valid else np.nan,
            }
        )

    return rows


def main():
    print("=" * 78)
    print("SESSION-AWARE DYNAMIC TRIPLE-BARRIER YEARLY DIAGNOSTIC")
    print("=" * 78)
    print("Horizon: 60m")
    print("Multipliers: 0.50x, 0.75x, 1.00x, 1.25x, 1.50x")
    print("2026: diagnostic only")
    print("=" * 78)

    all_rows = []

    for multiplier, path in FILES.items():
        print("\n" + "-" * 78)
        print(f"Reading {multiplier:.2f}x: {path}")
        print("-" * 78)
        all_rows.extend(process_file(path, multiplier))

    df = pd.DataFrame(all_rows).sort_values(
        ["multiplier", "year"]
    ).reset_index(drop=True)

    pct_sum = (
        df["short_pct_valid"]
        + df["timeout_pct_valid"]
        + df["long_pct_valid"]
    )
    if not np.allclose(
        pct_sum.dropna().to_numpy(), 100.0, atol=1e-6
    ):
        raise AssertionError("Valid class percentages do not sum to 100%.")

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR VALID CLASS DISTRIBUTIONS")
    print("=" * 78)

    for multiplier in FILES:
        print(f"\n--- {multiplier:.2f}x ---")
        sub = df[df["multiplier"] == multiplier]
        print(
            sub[
                [
                    "year",
                    "short_pct_valid",
                    "timeout_pct_valid",
                    "long_pct_valid",
                    "valid_rows",
                    "session_closed_rows",
                    "invalid_rows",
                    "ambiguous_rows",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

    # Summaries for the model-development years only (2019-2025).
    dev = df[df["year"].between(2019, 2025)].copy()

    stability_rows = []
    for multiplier in FILES:
        sub = dev[dev["multiplier"] == multiplier]

        stability_rows.append(
            {
                "multiplier": multiplier,
                "mean_short_pct_2019_2025": sub["short_pct_valid"].mean(),
                "std_short_pct_2019_2025": sub["short_pct_valid"].std(ddof=1),
                "mean_timeout_pct_2019_2025": sub["timeout_pct_valid"].mean(),
                "std_timeout_pct_2019_2025": sub["timeout_pct_valid"].std(ddof=1),
                "mean_long_pct_2019_2025": sub["long_pct_valid"].mean(),
                "std_long_pct_2019_2025": sub["long_pct_valid"].std(ddof=1),
                "min_valid_rows": int(sub["valid_rows"].min()),
            }
        )

    stability = pd.DataFrame(stability_rows)

    print("\n" + "=" * 78)
    print("2019-2025 DISTRIBUTION STABILITY")
    print("=" * 78)
    print(
        stability.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    df.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)
    print(f"Saved: {OUT}")
    print("2026 remains diagnostic only.")
    print("=" * 78)


if __name__ == "__main__":
    main()
