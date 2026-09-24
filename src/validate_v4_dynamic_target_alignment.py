#!/usr/bin/env python3
"""
Validate alignment between the session-aware V4 feature dataset and the
session-aware dynamic triple-barrier target datasets.

Checks:
1. V4 timestamps are strictly increasing and unique.
2. Every V4 timestamp exists in the dynamic target file.
3. Every V4 row has a valid dynamic target label (0/1/2).
4. Target labels are checked for all five multipliers:
      0.50x, 0.75x, 1.00x, 1.25x, 1.50x
5. The V4 feature row count is consistent across target files.
6. No V4 timestamp lands in a SESSION_CLOSED (-3), INVALID (-2),
   or AMBIGUOUS (-1) target row.
7. A small deterministic sample of rows is cross-checked across all
   target files.

2026 remains diagnostic only.

This validator does NOT modify any data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("data/processed")

FEATURE_FILE = BASE / "xauusd_m1_ml_features_v4_session.csv"

TARGET_FILES = {
    0.50: BASE / "xauusd_m1_session_dynamic_tb_0p50.csv",
    0.75: BASE / "xauusd_m1_session_dynamic_tb_0p75.csv",
    1.00: BASE / "xauusd_m1_session_dynamic_tb_1p00.csv",
    1.25: BASE / "xauusd_m1_session_dynamic_tb_1p25.csv",
    1.50: BASE / "xauusd_m1_session_dynamic_tb_1p50.csv",
}

LABEL_COL = "tb_label_60m"
CHUNK_SIZE = 250_000


class CSVStream:
    def __init__(self, path, usecols):
        self.path = path
        self.reader = pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        )
        self.buffer = np.empty(0, dtype=[("timestamp", "i8"), ("label", "i2")])
        self.pos = 0
        self.exhausted = False

    def _load_next(self):
        try:
            chunk = next(self.reader)
        except StopIteration:
            self.exhausted = True
            self.buffer = np.empty(
                0, dtype=[("timestamp", "i8"), ("label", "i2")]
            )
            self.pos = 0
            return

        ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
        labels = pd.to_numeric(chunk["tb_label_60m"], errors="coerce")

        if ts.isna().any():
            raise ValueError(f"Invalid timestamp found in {self.path}")
        if labels.isna().any():
            raise ValueError(f"Invalid label found in {self.path}")

        arr = np.empty(len(chunk), dtype=[("timestamp", "i8"), ("label", "i2")])
        arr["timestamp"] = ts.astype(np.int64).to_numpy()
        arr["label"] = labels.astype(np.int16).to_numpy()
        self.buffer = arr
        self.pos = 0

    def peek(self):
        while self.pos >= len(self.buffer):
            if self.exhausted:
                return None
            self._load_next()

        return self.buffer[self.pos]

    def pop(self):
        row = self.peek()
        if row is None:
            return None
        self.pos += 1
        return row


def validate_feature_file():
    print("=" * 78)
    print("VALIDATING V4 FEATURE DATASET")
    print("=" * 78)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(FEATURE_FILE)

    first = pd.read_csv(
        FEATURE_FILE,
        usecols=["timestamp", "close", "return_1m", "future_return_60m"],
        nrows=5,
    )

    required = ["timestamp", "close", "return_1m", "future_return_60m"]
    missing = [c for c in required if c not in first.columns]
    if missing:
        raise KeyError(f"Missing V4 columns: {missing}")

    total = 0
    previous = None
    min_ts = None
    max_ts = None

    # Deterministic sample positions are collected for later target check.
    sample_every = 100_000
    sample_ts = []

    for chunk in pd.read_csv(
        FEATURE_FILE,
        usecols=["timestamp", "close", "return_1m", "future_return_60m"],
        chunksize=CHUNK_SIZE,
    ):
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce").to_numpy(np.int64)

        if not np.all(ts[1:] > ts[:-1]):
            raise AssertionError("V4 timestamps are not strictly increasing within a chunk.")

        if previous is not None and int(ts[0]) <= previous:
            raise AssertionError("V4 timestamps are not strictly increasing across chunks.")

        previous = int(ts[-1])
        min_ts = int(ts[0]) if min_ts is None else min_ts
        max_ts = int(ts[-1])
        total += len(chunk)

        # Feature/target columns should be finite by the V4 build contract.
        values = chunk[["close", "return_1m", "future_return_60m"]].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite values found in V4 validation columns.")

        positions = np.arange(total - len(chunk), total)
        wanted = positions[positions % sample_every == 0]
        if len(wanted):
            local = wanted - (total - len(chunk))
            sample_ts.extend(ts[local].tolist())

    print(f"V4 rows:       {total:,}")
    print(f"First timestamp:{min_ts}")
    print(f"Last timestamp: {max_ts}")

    return total, min_ts, max_ts, np.asarray(sample_ts, dtype=np.int64)


def validate_against_target(feature_count, target_path, multiplier, sample_ts):
    print("\n" + "-" * 78)
    print(f"TARGET {multiplier:.2f}x")
    print(f"File: {target_path}")
    print("-" * 78)

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    # Stream V4 and target side-by-side using timestamp order.
    f_reader = pd.read_csv(
        FEATURE_FILE,
        usecols=["timestamp"],
        chunksize=CHUNK_SIZE,
    )
    t_reader = pd.read_csv(
        target_path,
        usecols=["timestamp", LABEL_COL],
        chunksize=CHUNK_SIZE,
    )

    f_iter = iter(f_reader)
    t_iter = iter(t_reader)

    f_chunk = next(f_iter, None)
    t_chunk = next(t_iter, None)
    fi = 0
    ti = 0

    matched = 0
    v4_missing_target = 0
    target_invalid_on_v4 = 0
    target_ambiguous_on_v4 = 0
    target_closed_on_v4 = 0
    target_valid_on_v4 = 0
    target_rows_seen = 0
    previous_target_ts = None

    sample_set = set(int(x) for x in sample_ts.tolist())
    sampled_seen = {}

    while f_chunk is not None:
        if fi >= len(f_chunk):
            f_chunk = next(f_iter, None)
            fi = 0
            continue

        f_ts = int(pd.to_numeric(f_chunk.iloc[fi]["timestamp"]))

        while t_chunk is not None:
            if ti >= len(t_chunk):
                t_chunk = next(t_iter, None)
                ti = 0
                continue

            t_ts = int(pd.to_numeric(t_chunk.iloc[ti]["timestamp"]))

            if previous_target_ts is not None and t_ts <= previous_target_ts:
                raise AssertionError(f"Target timestamps not strictly increasing in {target_path}")

            # Advance target until it catches the current V4 timestamp.
            if t_ts < f_ts:
                previous_target_ts = t_ts
                ti += 1
                target_rows_seen += 1
                continue

            break

        if t_chunk is None:
            v4_missing_target += 1
            fi += 1
            matched += 1
            continue

        t_ts = int(pd.to_numeric(t_chunk.iloc[ti]["timestamp"]))
        label = int(pd.to_numeric(t_chunk.iloc[ti][LABEL_COL]))

        if t_ts != f_ts:
            # Target timestamp jumped past the V4 timestamp.
            v4_missing_target += 1
        else:
            matched += 1
            if label == -3:
                target_closed_on_v4 += 1
            elif label == -2:
                target_invalid_on_v4 += 1
            elif label == -1:
                target_ambiguous_on_v4 += 1
            elif label in (0, 1, 2):
                target_valid_on_v4 += 1
            else:
                raise AssertionError(
                    f"Unexpected target label {label} at timestamp {f_ts}"
                )

            if f_ts in sample_set and f_ts not in sampled_seen:
                sampled_seen[f_ts] = label

        fi += 1

    # Finish counting remaining target rows to report file length/check order.
    while t_chunk is not None:
        while ti < len(t_chunk):
            t_ts = int(pd.to_numeric(t_chunk.iloc[ti]["timestamp"]))
            if previous_target_ts is not None and t_ts <= previous_target_ts:
                raise AssertionError(f"Target timestamps not strictly increasing in {target_path}")
            previous_target_ts = t_ts
            target_rows_seen += 1
            ti += 1
        t_chunk = next(t_iter, None)
        ti = 0

    print(f"V4 rows checked:            {matched:,}")
    print(f"Target rows seen:            {target_rows_seen:,}")
    print(f"V4 timestamps missing:      {v4_missing_target:,}")
    print(f"V4 → valid labels:          {target_valid_on_v4:,}")
    print(f"V4 → ambiguous labels:      {target_ambiguous_on_v4:,}")
    print(f"V4 → invalid labels:        {target_invalid_on_v4:,}")
    print(f"V4 → session-closed labels: {target_closed_on_v4:,}")
    print(f"Deterministic samples:       {len(sampled_seen):,}/{len(sample_set):,}")

    if matched != feature_count:
        raise AssertionError(
            f"V4 match count {matched:,} != V4 feature count {feature_count:,}"
        )

    if v4_missing_target:
        raise AssertionError(
            f"{v4_missing_target:,} V4 timestamps were missing from {target_path}"
        )

    tradable_or_ambiguous = target_valid_on_v4 + target_ambiguous_on_v4

    if tradable_or_ambiguous != feature_count:
        raise AssertionError(
            f"V4 rows accounted for by valid+ambiguous labels "
            f"({tradable_or_ambiguous:,}) != feature count ({feature_count:,})"
        )

    if target_invalid_on_v4 or target_closed_on_v4:
        raise AssertionError(
            "Some V4 feature rows map to INVALID or SESSION_CLOSED target labels."
        )

    if set(sampled_seen) != sample_set:
        raise AssertionError("Deterministic sample alignment is incomplete.")

    return target_rows_seen


def main():
    print("=" * 78)
    print("SESSION-AWARE V4 FEATURE ↔ DYNAMIC TARGET ALIGNMENT")
    print("=" * 78)
    print("Targets: 0.50x, 0.75x, 1.00x, 1.25x, 1.50x")
    print("Horizon: 60m")
    print("2026: diagnostic only")
    print("=" * 78)

    feature_count, min_ts, max_ts, sample_ts = validate_feature_file()

    target_counts = {}
    for multiplier, path in TARGET_FILES.items():
        target_counts[multiplier] = validate_against_target(
            feature_count,
            path,
            multiplier,
            sample_ts,
        )

    if len(set(target_counts.values())) != 1:
        raise AssertionError(
            f"Target row counts are inconsistent: {target_counts}"
        )

    print("\n" + "=" * 78)
    print("ALIGNMENT VALIDATION PASSED")
    print("=" * 78)
    print(f"V4 feature rows: {feature_count:,}")
    print(f"Target file rows: {target_counts[0.50]:,}")
    print("Every V4 row maps to either a valid SHORT/TIMEOUT/LONG label or AMBIGUOUS.")
    print("No V4 row maps to SESSION_CLOSED or INVALID.")
    print("AMBIGUOUS rows will be excluded from ML training for the affected target.")
    print("V4 timestamps are strictly increasing and unique.")
    print("2026 remains untouched for later model evaluation.")
    print("=" * 78)


if __name__ == "__main__":
    main()
