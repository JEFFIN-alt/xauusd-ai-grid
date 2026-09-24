#!/usr/bin/env python3
"""
Build and audit a session-aware mask for the XAUUSD canonical M1 dataset.

Idea:
- The canonical file contains long stretches of unchanged CLOSE prices that
  represent closed-market/placeholder periods.
- Short flat runs can occur in real market data, so we do NOT remove every
  flat candle.
- For this first session filter, a consecutive identical-close run of
  >= 60 minutes is treated as a CLOSED/INACTIVE RUN.
- This script does NOT modify the canonical OHLC dataset.
- It creates a row-level mask and a diagnostic summary.

It also creates "active segments" split at:
  1) existing timestamp gaps, OR
  2) long flat-close runs (>= 60 minutes)

These active segments will later be used so feature windows and dynamic
volatility targets do not cross closed-market periods.

2026 is diagnostic only.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("data/processed")
CANONICAL = BASE / "xauusd_m1_2019_2026_canonical.csv"

MASK_OUT = BASE / "xauusd_m1_session_mask.csv"
SUMMARY_OUT = BASE / "xauusd_m1_session_filter_summary.csv"
SEGMENTS_OUT = BASE / "xauusd_m1_active_segments.csv"

CHUNK_SIZE = 500_000
EXPECTED_MS = 60_000
CLOSED_FLAT_MINUTES = 60


def load_canonical():
    print("=" * 78)
    print("Loading canonical XAUUSD M1...")
    print("=" * 78)

    parts = []
    for chunk in pd.read_csv(
        CANONICAL,
        usecols=["timestamp", "open", "high", "low", "close"],
        chunksize=CHUNK_SIZE,
    ):
        parts.append(chunk)

    df = pd.concat(parts, ignore_index=True)
    del parts

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(
        dtype=np.int64
    )
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(
        dtype=np.float64
    )

    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    if not np.isfinite(close).all():
        raise ValueError("Canonical close contains non-finite values.")

    print(f"Rows: {len(ts):,}")
    return ts, open_, high, low, close


def find_base_segments(ts):
    delta = np.diff(ts)
    breaks = np.flatnonzero(delta != EXPECTED_MS) + 1
    starts = np.r_[0, breaks].astype(np.int64)
    ends = np.r_[breaks, len(ts)].astype(np.int64)
    return starts, ends


def find_flat_runs(close, starts, ends):
    """
    Return run_length_per_row and a boolean closed_run mask.

    A run is marked closed only if:
      - close stays exactly unchanged
      - run length >= CLOSED_FLAT_MINUTES
    """
    n = len(close)
    run_length = np.ones(n, dtype=np.int32)
    closed_mask = np.zeros(n, dtype=bool)

    run_records = []

    for s, e in zip(starts, ends):
        s = int(s)
        e = int(e)
        i = s

        while i < e:
            j = i + 1
            while j < e and close[j] == close[i]:
                j += 1

            length = j - i
            run_length[i:j] = length

            if length >= CLOSED_FLAT_MINUTES:
                closed_mask[i:j] = True
                run_records.append((i, j, length))

            i = j

    return run_length, closed_mask, run_records


def build_active_segments(ts, closed_mask, base_starts, base_ends):
    """
    Split base continuous segments further around closed rows.

    A segment is ACTIVE only if all its rows are outside closed_mask.
    """
    active_starts = []
    active_ends = []

    for base_s, base_e in zip(base_starts, base_ends):
        s = int(base_s)
        e = int(base_e)

        if s >= e:
            continue

        local_closed = closed_mask[s:e]

        # Start a new active run after every closed row.
        cuts = np.flatnonzero(local_closed) + s

        current = s
        for c in cuts:
            c = int(c)
            if current < c:
                active_starts.append(current)
                active_ends.append(c)
            current = c + 1

        if current < e:
            active_starts.append(current)
            active_ends.append(e)

    return (
        np.asarray(active_starts, dtype=np.int64),
        np.asarray(active_ends, dtype=np.int64),
    )


def summarize_years(ts, closed_mask, run_length):
    years = pd.to_datetime(ts, unit="ms", utc=True).year.to_numpy()

    rows = []
    for year in range(2019, 2027):
        m = years == year
        closed = m & closed_mask

        rows.append(
            {
                "year": year,
                "total_rows": int(m.sum()),
                "closed_flat_rows_ge60": int(closed.sum()),
                "closed_flat_pct": (
                    float(closed.sum()) / max(1, int(m.sum())) * 100.0
                ),
                "active_rows": int((m & ~closed_mask).sum()),
                "flat_rows_ge2": int((m & (run_length >= 2)).sum()),
                "flat_rows_ge60": int((m & (run_length >= 60)).sum()),
                "flat_rows_ge120": int((m & (run_length >= 120)).sum()),
            }
        )

    return pd.DataFrame(rows)


def make_mask_df(ts, run_length, closed_mask):
    return pd.DataFrame(
        {
            "timestamp": ts,
            "close_flat_run_minutes": run_length,
            "closed_market_mask": closed_mask.astype(np.uint8),
            "active_market_mask": (~closed_mask).astype(np.uint8),
        }
    )


def main():
    print("=" * 78)
    print("XAUUSD SESSION-AWARE MARKET FILTER")
    print("=" * 78)
    print(
        f"Closed-market proxy: consecutive identical CLOSE run >= "
        f"{CLOSED_FLAT_MINUTES} minutes"
    )
    print("Canonical OHLC will NOT be modified.")
    print("2026: diagnostic only.")
    print("=" * 78)

    ts, open_, high, low, close = load_canonical()

    base_starts, base_ends = find_base_segments(ts)
    print(f"Existing timestamp-continuous segments: {len(base_starts):,}")

    run_length, closed_mask, run_records = find_flat_runs(
        close, base_starts, base_ends
    )

    print(f"Detected closed/inactive flat runs: {len(run_records):,}")
    print(f"Closed-mask rows: {int(closed_mask.sum()):,}")
    print(
        f"Closed-mask percentage: "
        f"{closed_mask.sum() / len(closed_mask) * 100:.2f}%"
    )

    active_starts, active_ends = build_active_segments(
        ts, closed_mask, base_starts, base_ends
    )

    lengths = active_ends - active_starts
    print(f"Active segments after filter: {len(active_starts):,}")
    if len(lengths):
        print(f"Shortest active segment: {int(lengths.min()):,} rows")
        print(f"Median active segment: {int(np.median(lengths)):,} rows")
        print(f"Longest active segment: {int(lengths.max()):,} rows")

    yearly = summarize_years(ts, closed_mask, run_length)

    print("\n" + "=" * 78)
    print("YEAR-BY-YEAR SESSION FILTER")
    print("=" * 78)
    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # Save row mask.
    mask_df = make_mask_df(ts, run_length, closed_mask)
    mask_df.to_csv(MASK_OUT, index=False)

    # Save yearly summary.
    yearly.to_csv(SUMMARY_OUT, index=False)

    # Save active segments.
    seg_df = pd.DataFrame(
        {
            "active_segment_id": np.arange(len(active_starts), dtype=np.int64),
            "start_index": active_starts,
            "end_index_exclusive": active_ends,
            "length_minutes": active_ends - active_starts,
            "start_timestamp": pd.to_datetime(
                ts[active_starts], unit="ms", utc=True
            ).astype(str),
            "end_timestamp_exclusive": pd.to_datetime(
                ts[np.minimum(active_ends, len(ts) - 1)],
                unit="ms",
                utc=True,
            ).astype(str),
        }
    )
    seg_df.to_csv(SEGMENTS_OUT, index=False)

    # Cross-check against the earlier zero-vol diagnostic concept.
    print("\n" + "=" * 78)
    print("SESSION FILTER CHECKS")
    print("=" * 78)

    # Count closed rows among the previously diagnosed long flat rows.
    flat60 = run_length >= CLOSED_FLAT_MINUTES
    print(
        f"Rows in flat runs >= 60m: {int(flat60.sum()):,}"
    )
    print(
        f"Rows in flat runs >= 60m also marked closed: "
        f"{int((flat60 & closed_mask).sum()):,}"
    )

    # Check for any active run long enough to support a 60m historical
    # volatility window + 60m future horizon.
    sufficient_active = lengths >= 121
    print(
        f"Active segments >= 121 rows (enough for 60m history + 60m future): "
        f"{int(sufficient_active.sum()):,}/{len(lengths):,}"
    )

    # Show largest closed runs.
    if run_records:
        top = sorted(run_records, key=lambda x: x[2], reverse=True)[:20]
        print("\nTop closed flat runs:")
        print("start_index,end_index_exclusive,length_minutes,start_timestamp")
        for s, e, length in top:
            print(
                f"{s},{e},{length},"
                f"{pd.to_datetime(ts[s], unit='ms', utc=True).isoformat()}"
            )

    print("\n" + "=" * 78)
    print("SESSION FILTER COMPLETE")
    print("=" * 78)
    print(f"Saved row mask:       {MASK_OUT}")
    print(f"Saved yearly summary: {SUMMARY_OUT}")
    print(f"Saved active segments: {SEGMENTS_OUT}")
    print("No original OHLC data was modified.")
    print("2026 remains diagnostic only.")
    print("=" * 78)


if __name__ == "__main__":
    main()
