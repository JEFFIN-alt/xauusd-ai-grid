#!/usr/bin/env python3
"""
SESSION-AWARE V4 FEATURE DATASET — EXACT V3 FEATURE FORMULAS

This builder reproduces the existing V3 feature definitions exactly, with
one intentional methodological change:

    V3 segmentation:
        split only at timestamp gaps

    V4 segmentation:
        split at timestamp gaps AND session-closed flat runs (>= 60 min)

Therefore no backward feature, rolling window, or future target can cross a
closed-market period.

Source:
    data/processed/xauusd_m1_2019_2026_canonical.csv

Session boundaries:
    data/processed/xauusd_m1_active_segments.csv

Output:
    data/processed/xauusd_m1_ml_features_v4_session.csv

42 features:
    return_1m, return_5m, return_15m, return_30m, return_60m
    candle_range, body, abs_body, upper_wick, lower_wick
    body_range_ratio, upper_wick_ratio, lower_wick_ratio
    volatility_15m, volatility_30m, volatility_60m, volatility_240m
    momentum_5m, momentum_15m, momentum_30m, momentum_60m
    sma_15, sma_60, sma_240
    distance_sma_15, distance_sma_60, distance_sma_240
    sma_15_slope, sma_60_slope, sma_240_slope
    rolling_high_60, rolling_low_60, rolling_high_240, rolling_low_240
    distance_high_60, distance_low_60, distance_high_240, distance_low_240
    hour_sin, hour_cos, minute_sin, minute_cos

4 targets:
    future_return_5m, future_return_15m, future_return_30m, future_return_60m

Important:
- V3 formulas are preserved, including momentum as CLOSE difference,
  cyclical minute features using minute/60, candle flat-ratio handling,
  rolling std default ddof=1, and the existing slope definitions.
- V4 only changes the segmentation boundary.
- 2026 is included in the dataset for diagnostics only and must remain
  excluded from later model selection/testing.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

BASE = Path("data/processed")
CANONICAL = BASE / "xauusd_m1_2019_2026_canonical.csv"
SEGMENTS_FILE = BASE / "xauusd_m1_active_segments.csv"
OUTPUT = BASE / "xauusd_m1_ml_features_v4_session.csv"

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

TARGETS = [
    "future_return_5m",
    "future_return_15m",
    "future_return_30m",
    "future_return_60m",
]

BASE_COLUMNS = ["timestamp", "open", "high", "low", "close"]
OUTPUT_COLUMNS = BASE_COLUMNS + FEATURES + TARGETS

# Exact V3 maximum backward dependency:
# sma_240 needs 240 rows, and sma_240_slope compares current SMA_240
# to SMA_240 shifted by another 240 rows => first valid row is 479.
MAX_BACKWARD_BARS = 480
MAX_FORWARD_BARS = 60
MIN_SEGMENT_LENGTH = MAX_BACKWARD_BARS + MAX_FORWARD_BARS


def load_segments(n_rows: int):
    if not SEGMENTS_FILE.exists():
        raise FileNotFoundError(SEGMENTS_FILE)

    seg = pd.read_csv(SEGMENTS_FILE)

    required = ["start_index", "end_index_exclusive", "length_minutes"]
    missing = [c for c in required if c not in seg.columns]
    if missing:
        raise KeyError(f"Missing segment columns: {missing}")

    starts = (
        pd.to_numeric(seg["start_index"], errors="coerce")
        .astype(np.int64)
        .to_numpy()
    )
    ends = (
        pd.to_numeric(seg["end_index_exclusive"], errors="coerce")
        .astype(np.int64)
        .to_numpy()
    )

    if len(starts) == 0:
        raise ValueError("No active segments found.")

    if np.any(starts < 0) or np.any(ends > n_rows) or np.any(ends <= starts):
        raise ValueError("Invalid active segment bounds.")

    if not np.all(ends[:-1] <= starts[1:]):
        raise ValueError("Active segments overlap or are out of order.")

    lengths = ends - starts

    return starts, ends, lengths


def build_segment_features(working: pd.DataFrame) -> pd.DataFrame:
    """
    Exact feature formulas from build_ml_features_v3.py, applied to one
    session-aware active segment so nothing can cross a closed-market run.
    """
    working = working.copy()

    close = working["close"]
    high = working["high"]
    low = working["low"]
    open_price = working["open"]

    # ========================================================
    # BACKWARD RETURN FEATURES — exact V3 formula
    # ========================================================
    for horizon in [1, 5, 15, 30, 60]:
        previous_close = close.shift(horizon)
        working[f"return_{horizon}m"] = (
            close / previous_close - 1.0
        )

    # ========================================================
    # CANDLE FEATURES — exact V3 formula
    # ========================================================
    working["candle_range"] = high - low
    working["body"] = close - open_price
    working["abs_body"] = working["body"].abs()

    body_high = pd.concat([open_price, close], axis=1).max(axis=1)
    body_low = pd.concat([open_price, close], axis=1).min(axis=1)

    working["upper_wick"] = high - body_high
    working["lower_wick"] = body_low - low

    # ========================================================
    # CANDLE RATIOS — exact V3 formula
    # ========================================================
    safe_range = working["candle_range"].replace(0, np.nan)

    working["body_range_ratio"] = (
        working["abs_body"] / safe_range
    )
    working["upper_wick_ratio"] = (
        working["upper_wick"] / safe_range
    )
    working["lower_wick_ratio"] = (
        working["lower_wick"] / safe_range
    )

    flat = working["candle_range"] == 0
    working.loc[
        flat,
        ["body_range_ratio", "upper_wick_ratio", "lower_wick_ratio"],
    ] = 0.0

    # ========================================================
    # VOLATILITY — exact V3 formula
    # Pandas rolling std default ddof=1, matching V3.
    # ========================================================
    working["volatility_15m"] = (
        working["return_1m"].rolling(window=15, min_periods=15).std()
    )
    working["volatility_30m"] = (
        working["return_1m"].rolling(window=30, min_periods=30).std()
    )
    working["volatility_60m"] = (
        working["return_1m"].rolling(window=60, min_periods=60).std()
    )
    working["volatility_240m"] = (
        working["return_1m"].rolling(window=240, min_periods=240).std()
    )

    # ========================================================
    # MOMENTUM — exact V3 formula
    # IMPORTANT: raw CLOSE difference, NOT percentage return.
    # ========================================================
    for horizon in [5, 15, 30, 60]:
        previous_close = close.shift(horizon)
        working[f"momentum_{horizon}m"] = close - previous_close

    # ========================================================
    # MOVING AVERAGES — exact V3 formula
    # ========================================================
    working["sma_15"] = close.rolling(
        window=15, min_periods=15
    ).mean()
    working["sma_60"] = close.rolling(
        window=60, min_periods=60
    ).mean()
    working["sma_240"] = close.rolling(
        window=240, min_periods=240
    ).mean()

    # ========================================================
    # DISTANCE FROM SMA — exact V3 formula
    # ========================================================
    working["distance_sma_15"] = close / working["sma_15"] - 1.0
    working["distance_sma_60"] = close / working["sma_60"] - 1.0
    working["distance_sma_240"] = close / working["sma_240"] - 1.0

    # ========================================================
    # SMA SLOPE — exact V3 formula
    # ========================================================
    working["sma_15_slope"] = (
        working["sma_15"] - working["sma_15"].shift(15)
    )
    working["sma_60_slope"] = (
        working["sma_60"] - working["sma_60"].shift(60)
    )
    working["sma_240_slope"] = (
        working["sma_240"] - working["sma_240"].shift(240)
    )

    # ========================================================
    # ROLLING HIGH / LOW — exact V3 formula
    # ========================================================
    working["rolling_high_60"] = high.rolling(
        window=60, min_periods=60
    ).max()
    working["rolling_low_60"] = low.rolling(
        window=60, min_periods=60
    ).min()
    working["rolling_high_240"] = high.rolling(
        window=240, min_periods=240
    ).max()
    working["rolling_low_240"] = low.rolling(
        window=240, min_periods=240
    ).min()

    # ========================================================
    # DISTANCES FROM HIGH / LOW — exact V3 formula
    # ========================================================
    working["distance_high_60"] = (
        close / working["rolling_high_60"] - 1.0
    )
    working["distance_low_60"] = (
        close / working["rolling_low_60"] - 1.0
    )
    working["distance_high_240"] = (
        close / working["rolling_high_240"] - 1.0
    )
    working["distance_low_240"] = (
        close / working["rolling_low_240"] - 1.0
    )

    # ========================================================
    # TIME FEATURES — exact V3 formula
    # minute feature uses MINUTE OF HOUR / 60.
    # ========================================================
    datetime = pd.to_datetime(
        working["timestamp"],
        unit="ms",
        utc=True,
    )

    hour = datetime.dt.hour
    minute = datetime.dt.minute

    working["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    working["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    working["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    working["minute_cos"] = np.cos(2 * np.pi * minute / 60)

    # ========================================================
    # MULTI-HORIZON TARGETS — exact V3 outcome on an active segment
    #
    # Because this is one active segment, shift(-h) is equivalent to V3's
    # exact timestamp + same-segment validity check.
    # ========================================================
    for horizon in [5, 15, 30, 60]:
        future_close = close.shift(-horizon)
        working[f"future_return_{horizon}m"] = (
            future_close / close - 1.0
        )

    # V3's final valid rows require every feature/target to be non-NaN.
    valid = working[FEATURES + TARGETS].notna().all(axis=1)

    out = working.loc[valid, OUTPUT_COLUMNS].copy()

    return out


def main():
    print("=" * 78)
    print("SESSION-AWARE ML FEATURE DATASET V4")
    print("=" * 78)
    print("Feature formulas: EXACT V3 definitions")
    print("Only segmentation changes: timestamp gaps + closed-market runs")
    print("Output excludes rows inside closed runs.")
    print("2026 retained for diagnostics only.")
    print("=" * 78)

    if not CANONICAL.exists():
        raise FileNotFoundError(CANONICAL)

    t0 = time.time()

    print("\nLoading canonical OHLC...")
    parts = []
    for chunk in pd.read_csv(
        CANONICAL,
        usecols=BASE_COLUMNS,
        chunksize=500_000,
    ):
        parts.append(chunk)

    canonical = pd.concat(parts, ignore_index=True)
    del parts

    ts_num = pd.to_numeric(canonical["timestamp"], errors="coerce")
    if ts_num.isna().any():
        raise ValueError("Invalid timestamp found in canonical data.")

    canonical["timestamp"] = ts_num.astype(np.int64)

    for col in ["open", "high", "low", "close"]:
        canonical[col] = pd.to_numeric(
            canonical[col], errors="coerce"
        ).astype(np.float64)

    if canonical[BASE_COLUMNS[1:]].isna().any().any():
        raise ValueError("NaN OHLC values found.")

    if len(canonical) < 1:
        raise ValueError("Canonical dataset is empty.")

    ts = canonical["timestamp"].to_numpy(np.int64, copy=False)
    if not np.all(ts[1:] > ts[:-1]):
        raise ValueError("Canonical timestamps are not strictly increasing.")

    print(f"Canonical rows: {len(canonical):,}")

    starts, ends, lengths = load_segments(len(canonical))

    eligible = lengths >= MIN_SEGMENT_LENGTH
    print(f"Active segments: {len(starts):,}")
    print(
        f"Segments >= {MIN_SEGMENT_LENGTH} rows: "
        f"{int(eligible.sum()):,}"
    )
    print(
        f"Active rows represented: {int(lengths.sum()):,}"
    )

    # Fresh output. Keep ONE persistent UTF-8 text handle for the entire
    # dataset instead of opening/appending the file thousands of times.
    if OUTPUT.exists():
        OUTPUT.unlink()

    total_written = 0
    processed = 0
    header_written = False

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024,
    ) as out_fh:

        # Build/write segment-by-segment to keep memory bounded.
        for segment_id, (s, e, length) in enumerate(
            zip(starts, ends, lengths),
            start=1,
        ):
            if length < MIN_SEGMENT_LENGTH:
                continue

            seg = canonical.iloc[int(s):int(e)].copy()
            out_seg = build_segment_features(seg)

            if not out_seg.empty:
                out_seg.to_csv(
                    out_fh,
                    header=not header_written,
                    index=False,
                    lineterminator="\n",
                )
                header_written = True
                total_written += len(out_seg)

            processed += 1

            if processed % 100 == 0:
                out_fh.flush()
                print(
                    f"Eligible segments processed: {processed:,}/"
                    f"{int(eligible.sum()):,} | "
                    f"rows written: {total_written:,}"
                )

    if total_written == 0:
        raise RuntimeError("No V4 feature rows were produced.")

    # Validate output without loading it all.
    print("\nRunning output integrity checks...")

    # Binary-level safety check before pandas parses the file.
    raw_bytes = OUTPUT.read_bytes()
    nul_count = raw_bytes.count(b"\x00")
    if nul_count:
        bad_pos = raw_bytes.find(b"\x00")
        raise RuntimeError(
            f"Output contains {nul_count:,} NUL byte(s); first at byte {bad_pos}."
        )
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Output UTF-8 validation failed at byte position {exc.start}."
        ) from exc
    finally:
        del raw_bytes

    first = pd.read_csv(OUTPUT, nrows=5)
    missing = [c for c in OUTPUT_COLUMNS if c not in first.columns]
    if missing:
        raise AssertionError(f"Missing output columns: {missing}")

    if list(first.columns) != OUTPUT_COLUMNS:
        raise AssertionError(
            "Output column order does not match expected V4 schema."
        )

    rows_checked = 0
    prev_ts = None
    min_ts = None
    max_ts = None

    for chunk in pd.read_csv(OUTPUT, usecols=OUTPUT_COLUMNS, chunksize=250_000):
        ts_chunk = chunk["timestamp"].to_numpy(np.int64, copy=False)
        if len(ts_chunk) > 1 and np.any(ts_chunk[1:] <= ts_chunk[:-1]):
            raise AssertionError("Output timestamps are not strictly increasing.")

        if prev_ts is not None and int(ts_chunk[0]) <= prev_ts:
            raise AssertionError("Output timestamp order breaks between chunks.")

        vals = chunk[FEATURES + TARGETS].to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise AssertionError("Non-finite feature/target value found.")

        prev_ts = int(ts_chunk[-1])
        min_ts = int(ts_chunk[0]) if min_ts is None else min_ts
        max_ts = int(ts_chunk[-1])
        rows_checked += len(chunk)

    if rows_checked != total_written:
        raise AssertionError(
            f"Written row count mismatch: expected {total_written}, got {rows_checked}"
        )

    print("\n" + "=" * 78)
    print("SESSION-AWARE V4 FEATURE BUILD COMPLETE")
    print("=" * 78)
    print(f"Canonical rows:             {len(canonical):,}")
    print(f"Active segments:            {len(starts):,}")
    print(f"Eligible segments:          {int(eligible.sum()):,}")
    print(f"Feature-valid output rows:  {total_written:,}")
    print(f"Columns:                    {len(OUTPUT_COLUMNS)}")
    print(f"First timestamp:             {min_ts}")
    print(f"Last timestamp:              {max_ts}")
    print(f"Output:                     {OUTPUT}")
    print(f"Wall time:                   {time.time() - t0:.1f} sec")
    print("No canonical OHLC data was modified.")
    print("2026 remains included only for diagnostics and must stay untouched.")
    print("=" * 78)


if __name__ == "__main__":
    main()
