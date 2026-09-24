#!/usr/bin/env python3
"""
Diagnose INVALID rows in the dynamic volatility-adjusted triple-barrier targets.

Goal:
Explain why the dynamic target generator produced the same 585,966 INVALID
rows for every multiplier.

The diagnostic uses the canonical M1 data and reproduces the ENTRY-time
volatility calculation used by the dynamic target generator.

Invalid reasons are separated into:
  1. structural_invalid
       - insufficient 60-minute history inside a continuous segment
       - insufficient 60-minute future horizon inside a segment
  2. zero_or_nonfinite_volatility
       - previous 60 returns have sigma <= 0 or non-finite
  3. invalid_entry_price
       - entry close is non-finite/non-positive
  4. valid/ambiguous
       - used as a cross-check against generated dynamic target files

It reports totals and year-by-year counts.
2026 is diagnostic only.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

BASE = Path("data/processed")
CANONICAL = BASE / "xauusd_m1_2019_2026_canonical.csv"

DYNAMIC_FILES = {
    0.50: BASE / "xauusd_m1_dynamic_tb_0p50.csv",
    0.75: BASE / "xauusd_m1_dynamic_tb_0p75.csv",
    1.00: BASE / "xauusd_m1_dynamic_tb_1p00.csv",
    1.25: BASE / "xauusd_m1_dynamic_tb_1p25.csv",
    1.50: BASE / "xauusd_m1_dynamic_tb_1p50.csv",
}

HORIZON = 60
VOL_LOOKBACK = 60
EXPECTED_MS = 60_000


def load_canonical():
    print("=" * 78)
    print("Loading canonical M1 data...")
    print("=" * 78)
    parts = []
    for chunk in pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "close"],
        chunksize=500_000,
    ):
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    del parts

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(
        dtype=np.int64
    )
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    print(f"Rows: {len(ts):,}")
    print("Timestamp unit: epoch-ms")
    return ts, close


def get_segments(ts):
    breaks = np.flatnonzero(np.diff(ts) != EXPECTED_MS) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(ts)]
    return starts, ends


def diagnose(ts, close):
    n = len(ts)
    years = pd.to_datetime(ts, unit="ms", utc=True).year.to_numpy()

    structural = np.zeros(n, dtype=bool)
    zero_vol = np.zeros(n, dtype=bool)
    bad_price = np.zeros(n, dtype=bool)
    valid = np.zeros(n, dtype=bool)

    starts, ends = get_segments(ts)

    # Dynamic target generator uses only history strictly before entry.
    # To avoid crossing a segment boundary, require 60 complete returns
    # immediately before the entry.
    for s, e in zip(starts, ends):
        first = int(s + VOL_LOOKBACK)
        last = int(e - HORIZON - 1)

        if first > last:
            structural[s:e] = True
            continue

        if s < first:
            structural[s:first] = True
        if last + 1 < e:
            structural[last + 1:e] = True

        for i in range(first, last + 1):
            entry = close[i]

            if not np.isfinite(entry) or entry <= 0:
                bad_price[i] = True
                continue

            hist = close[i - VOL_LOOKBACK:i]
            if len(hist) != VOL_LOOKBACK or np.any(~np.isfinite(hist)):
                zero_vol[i] = True
                continue

            if np.any(hist <= 0):
                zero_vol[i] = True
                continue

            # 60 one-minute log returns ending at i-1.
            prev = hist[:-1]
            curr = hist[1:]
            rets = np.log(curr / prev)

            # Need 59 returns from the 60 closes above, plus the return
            # from close[i-60] to close[i-59] etc. To exactly reproduce
            # the generator's previous-60-return definition, construct
            # from close[i-61:i].
            hist2 = close[i - VOL_LOOKBACK - 1:i]
            if len(hist2) != VOL_LOOKBACK + 1 or np.any(hist2 <= 0) or np.any(~np.isfinite(hist2)):
                zero_vol[i] = True
                continue

            rets = np.log(hist2[1:] / hist2[:-1])
            sigma = float(np.std(rets, ddof=1))

            if not np.isfinite(sigma) or sigma <= 0.0:
                zero_vol[i] = True
                continue

            valid[i] = True

    return years, starts, ends, structural, zero_vol, bad_price, valid


def print_reason_summary(years, structural, zero_vol, bad_price, valid):
    print("\n" + "=" * 78)
    print("INVALID-ROW REASON SUMMARY")
    print("=" * 78)

    for name, mask in [
        ("structural_invalid", structural),
        ("zero_or_nonfinite_volatility", zero_vol),
        ("invalid_entry_price", bad_price),
        ("usable_dynamic_entry", valid),
    ]:
        print(f"{name:30s}: {int(mask.sum()):,}")

    overlap = (
        structural.astype(np.int8)
        + zero_vol.astype(np.int8)
        + bad_price.astype(np.int8)
        + valid.astype(np.int8)
    )
    if not np.all(overlap == 1):
        # Rows can be in none only if something unexpected happened.
        missing = int((overlap == 0).sum())
        overlap_rows = int((overlap > 1).sum())
        print(f"WARNING: unclassified rows: {missing:,}; overlapping: {overlap_rows:,}")

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR")
    print("=" * 78)

    rows = []
    for year in range(2019, 2027):
        m = years == year
        rows.append(
            {
                "year": year,
                "total_rows": int(m.sum()),
                "structural_invalid": int((m & structural).sum()),
                "zero_or_nonfinite_volatility": int((m & zero_vol).sum()),
                "invalid_entry_price": int((m & bad_price).sum()),
                "usable_dynamic_entry": int((m & valid).sum()),
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def cross_check_generated_files(ts, years):
    print("\n" + "=" * 78)
    print("CROSS-CHECK GENERATED DYNAMIC FILES")
    print("=" * 78)

    rows = []

    for mult, path in DYNAMIC_FILES.items():
        if not path.exists():
            print(f"Missing: {path}")
            continue

        head = pd.read_csv(path, nrows=2)
        cols = [
            "timestamp",
            "volatility_60m",
            "barrier_return_60m",
            "tb_label_60m",
        ]
        missing = [c for c in cols if c not in head.columns]
        if missing:
            print(f"{path}: missing {missing}")
            continue

        counts = {
            "multiplier": mult,
            "total_rows": 0,
            "invalid_label": 0,
            "ambiguous_label": 0,
            "valid_label": 0,
            "zero_or_nan_vol_rows": 0,
        }

        for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000):
            labels = pd.to_numeric(
                chunk["tb_label_60m"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            vol = pd.to_numeric(
                chunk["volatility_60m"], errors="coerce"
            ).to_numpy(dtype=np.float64)

            counts["total_rows"] += len(chunk)
            counts["invalid_label"] += int((labels == -2).sum())
            counts["ambiguous_label"] += int((labels == -1).sum())
            counts["valid_label"] += int(np.isin(labels, [0, 1, 2]).sum())
            counts["zero_or_nan_vol_rows"] += int(
                ((~np.isfinite(vol)) | (vol <= 0)).sum()
            )

        rows.append(counts)

    if rows:
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))


def main():
    print("=" * 78)
    print("DYNAMIC TRIPLE-BARRIER INVALID-ROW DIAGNOSTIC")
    print("=" * 78)
    print("Volatility: previous 60 continuous 1-minute returns")
    print("Horizon: 60 minutes")
    print("2026: diagnostic only")
    print("=" * 78)

    ts, close = load_canonical()

    print("\nFinding continuous segments...")
    starts, ends = get_segments(ts)
    lengths = ends - starts
    print(f"Segments: {len(starts):,}")
    print(
        f"Shortest segment: {int(lengths.min()):,} rows | "
        f"Median: {int(np.median(lengths)):,} | "
        f"Longest: {int(lengths.max()):,}"
    )
    print(
        f"Segments shorter than {VOL_LOOKBACK + HORIZON + 1} rows: "
        f"{int((lengths < VOL_LOOKBACK + HORIZON + 1).sum()):,}"
    )

    years, starts, ends, structural, zero_vol, bad_price, valid = diagnose(
        ts, close
    )

    yearly = print_reason_summary(
        years, structural, zero_vol, bad_price, valid
    )

    out = BASE / "xauusd_m1_dynamic_tb_invalid_diagnostic.csv"
    yearly.to_csv(out, index=False)
    print(f"\nSaved diagnostic: {out}")

    cross_check_generated_files(ts, years)

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
