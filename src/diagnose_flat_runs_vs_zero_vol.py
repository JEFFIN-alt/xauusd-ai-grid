#!/usr/bin/env python3
"""
Diagnose flat-close runs versus zero-volatility rows in the dynamic target setup.

Goal:
Determine whether the ~534k rows classified as zero/non-finite volatility
are explained by stretches of unchanged CLOSE prices.

Reports:
- total rows
- rows where close == previous close
- number and distribution of consecutive flat-close runs
- longest flat-close run
- rows belonging to flat runs of length >= 2/5/10/30/60
- year-by-year flat-run statistics
- exact overlap with the zero-volatility condition used by the dynamic target:
    std of previous 60 one-minute log returns == 0 or non-finite

2026 is diagnostic only.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("data/processed")
CANONICAL = BASE / "xauusd_m1_2019_2026_canonical.csv"
OUT = BASE / "xauusd_m1_flat_run_diagnostic.csv"

LOOKBACK = 60
EXPECTED_MS = 60_000


def load_data():
    parts = []
    for chunk in pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "close"],
        chunksize=500_000,
    ):
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.int64)
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    return ts, close


def find_segments(ts):
    breaks = np.flatnonzero(np.diff(ts) != EXPECTED_MS) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(ts)]
    return starts, ends


def flat_run_lengths(close, starts, ends):
    """
    Find consecutive identical CLOSE runs inside each continuous segment.
    Returns:
      run_length_per_row
      list of (start_index, end_index_exclusive, length)
    """
    n = len(close)
    run_len = np.ones(n, dtype=np.int32)
    runs = []

    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)
        i = s

        while i < e:
            j = i + 1
            while j < e and close[j] == close[i]:
                j += 1

            length = j - i
            run_len[i:j] = length

            if length >= 2:
                runs.append((i, j, length))

            i = j

    return run_len, runs


def zero_vol_mask(close, starts, ends):
    """
    Reproduce the dynamic target's previous-60-return volatility test.
    """
    n = len(close)
    zero_vol = np.zeros(n, dtype=bool)

    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)

        first = s + LOOKBACK
        last = e - 1

        for i in range(first, last + 1):
            hist = close[i - LOOKBACK - 1:i]

            if len(hist) != LOOKBACK + 1:
                zero_vol[i] = True
                continue

            if np.any(~np.isfinite(hist)) or np.any(hist <= 0):
                zero_vol[i] = True
                continue

            rets = np.log(hist[1:] / hist[:-1])
            sigma = np.std(rets, ddof=1)

            if not np.isfinite(sigma) or sigma <= 0.0:
                zero_vol[i] = True

    return zero_vol


def main():
    print("=" * 78)
    print("XAUUSD FLAT-CLOSE vs ZERO-VOLATILITY DIAGNOSTIC")
    print("=" * 78)

    ts, close = load_data()
    years = pd.to_datetime(ts, unit="ms", utc=True).year.to_numpy()

    starts, ends = find_segments(ts)
    run_len, runs = flat_run_lengths(close, starts, ends)

    print(f"Rows: {len(close):,}")
    print(f"Continuous segments: {len(starts):,}")

    flat_rows = int((run_len >= 2).sum())
    print(f"Rows inside flat-close runs (length >=2): {flat_rows:,}")
    print(f"Number of flat-close runs: {len(runs):,}")

    if runs:
        lengths = np.array([r[2] for r in runs], dtype=np.int32)
        print(f"Longest flat-close run: {int(lengths.max()):,} minutes")
        print(f"Median flat-close run: {float(np.median(lengths)):.1f} minutes")
        print(f"Mean flat-close run: {float(np.mean(lengths)):.1f} minutes")

        print("\nFlat-run thresholds:")
        for threshold in [2, 5, 10, 30, 60, 120]:
            rows = int((run_len >= threshold).sum())
            run_count = int((lengths >= threshold).sum())
            print(
                f"length >= {threshold:3d} min: "
                f"rows={rows:,} | runs={run_count:,}"
            )

    print("\nComputing exact zero-volatility mask...")
    zero_vol = zero_vol_mask(close, starts, ends)
    zc = int(zero_vol.sum())
    print(f"Zero/non-finite volatility rows: {zc:,}")

    overlap = zero_vol & (run_len >= 2)
    print(
        f"Zero-vol rows inside flat-close runs: {int(overlap.sum()):,} "
        f"({overlap.sum() / max(1, zc) * 100:.2f}% of zero-vol rows)"
    )

    not_flat = zero_vol & (run_len < 2)
    print(
        f"Zero-vol rows NOT inside flat-close runs: {int(not_flat.sum()):,} "
        f"({not_flat.sum() / max(1, zc) * 100:.2f}% of zero-vol rows)"
    )

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR")
    print("=" * 78)

    rows = []
    for year in range(2019, 2027):
        m = years == year
        z = m & zero_vol
        f2 = m & (run_len >= 2)
        f60 = m & (run_len >= 60)
        rows.append(
            {
                "year": year,
                "total_rows": int(m.sum()),
                "flat_run_rows_ge2": int(f2.sum()),
                "flat_run_rows_ge60": int(f60.sum()),
                "zero_vol_rows": int(z.sum()),
                "zero_vol_inside_flat_ge2": int((z & (run_len >= 2)).sum()),
                "zero_vol_pct_inside_flat_ge2":
                    float((z & (run_len >= 2)).sum()) / max(1, int(z.sum())) * 100,
            }
        )

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    out_df.to_csv(OUT, index=False)

    print("\nSaved:", OUT)

    if runs:
        top = sorted(runs, key=lambda x: x[2], reverse=True)[:20]
        print("\nTop 20 flat-close runs:")
        print("start_idx,end_idx_exclusive,length_minutes,start_timestamp")
        for s, e, length in top:
            print(
                f"{s},{e},{length},"
                f"{pd.to_datetime(ts[s], unit='ms', utc=True).isoformat()}"
            )

    print("\n" + "=" * 78)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
