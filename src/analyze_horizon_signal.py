import pandas as pd
import numpy as np
import lightgbm as lgb
import gc


FILE = "data/processed/xauusd_m1_15m_features.csv"

CHUNK_SIZE = 100_000

TRAIN_START = pd.Timestamp(
    "2019-01-01",
    tz="UTC"
)

TRAIN_END = pd.Timestamp(
    "2024-12-31 23:45:00",
    tz="UTC"
)

VALID_START = pd.Timestamp(
    "2025-01-01",
    tz="UTC"
)

VALID_END = pd.Timestamp(
    "2025-12-31 23:45:00",
    tz="UTC"
)

TARGET_15M = "future_return_15m"

HORIZONS = [5, 15, 30, 60]

TRAIN_SAMPLE = 500_000

RNG = np.random.default_rng(42)


print("=" * 70)
print("HORIZON SIGNAL DIAGNOSTIC")
print("=" * 70)


# --------------------------------------------------
# LOAD FEATURE NAMES
# --------------------------------------------------

columns = pd.read_csv(
    FILE,
    nrows=1
).columns.tolist()

excluded = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    TARGET_15M,
}

FEATURES = [
    c for c in columns
    if c not in excluded
]

print(f"\nFeatures: {len(FEATURES)}")


# --------------------------------------------------
# FIRST PASS
#
# We calculate future closes from the ORIGINAL
# chronological dataset.
# --------------------------------------------------

print("\nPass 1: calculating chronological targets...")


needed = [
    "timestamp",
    "close"
]


target_parts = []

carry = pd.DataFrame(
    columns=needed
)


for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=needed,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    # Add previous rows needed for future calculations.
    combined = pd.concat(
        [carry, chunk],
        ignore_index=True
    )

    timestamp = pd.to_datetime(
        combined["timestamp"],
        unit="ms",
        utc=True
    )

    combined["datetime"] = timestamp

    # Calculate targets using the actual chronological rows.
    for horizon in HORIZONS:

        future_close = (
            combined["close"]
            .shift(-horizon)
        )

        combined[
            f"target_{horizon}m"
        ] = (
            future_close /
            combined["close"]
        ) - 1

    # We only keep rows belonging to the current chunk.
    start_index = len(carry)

    current = combined.iloc[
        start_index:
    ].copy()

    # Keep only information needed later.
    target_columns = [
        "timestamp",
        "datetime",
        "close",
    ]

    for horizon in HORIZONS:
        target_columns.append(
            f"target_{horizon}m"
        )

    current = current[
        target_columns
    ]

    # Keep only train + validation period.
    mask = (
        (
            current["datetime"] >=
            TRAIN_START
        )
        &
        (
            current["datetime"] <=
            VALID_END
        )
    )

    current = current.loc[mask]

    if len(current):
        target_parts.append(
            current
        )

    # Carry enough rows into the next chunk.
    carry = combined.tail(
        max(HORIZONS)
    )[needed].copy()

    del chunk
    del combined
    del current

    if chunk_number % 10 == 0:
        print(
            f"Processed chunks: "
            f"{chunk_number}"
        )

    gc.collect()


targets = pd.concat(
    target_parts,
    ignore_index=True
)

del target_parts
del carry

gc.collect()


print(
    f"\nRows with targets: "
    f"{len(targets):,}"
)


# --------------------------------------------------
# SECOND PASS
#
# Load the 42 features and attach the already
# calculated chronological targets.
# --------------------------------------------------

print("\nPass 2: loading features...")


feature_usecols = [
    "timestamp"
] + FEATURES


feature_parts = []


for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=feature_usecols,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    dt = pd.to_datetime(
        chunk["timestamp"],
        unit="ms",
        utc=True
    )

    mask = (
        (dt >= TRAIN_START)
        &
        (dt <= VALID_END)
    )

    if mask.any():

        selected = chunk.loc[
            mask
        ].copy()

        feature_parts.append(
            selected
        )

    del chunk
    del dt

    if chunk_number % 10 == 0:
        print(
            f"Feature chunks: "
            f"{chunk_number}"
        )

    gc.collect()


features = pd.concat(
    feature_parts,
    ignore_index=True
)

del feature_parts

gc.collect()


# --------------------------------------------------
# MERGE
# --------------------------------------------------

print("\nCombining features and targets...")


data = features.merge(
    targets[
        [
            "timestamp"
        ]
        +
        [
            f"target_{h}m"
            for h in HORIZONS
        ]
    ],
    on="timestamp",
    how="inner",
    sort=False
)

del features
del targets

gc.collect()


print(
    f"Combined rows: "
    f"{len(data):,}"
)


# --------------------------------------------------
# TRAIN / VALIDATION MASKS
# --------------------------------------------------

dt = pd.to_datetime(
    data["timestamp"],
    unit="ms",
    utc=True
)

train_mask = (
    (dt >= TRAIN_START)
    &
    (dt <= TRAIN_END)
)

valid_mask = (
    (dt >= VALID_START)
    &
    (dt <= VALID_END)
)


train_full = data.loc[
    train_mask
].copy()

valid = data.loc[
    valid_mask
].copy()

del data
del dt

gc.collect()


# --------------------------------------------------
# FIXED TRAINING SAMPLE
# --------------------------------------------------

print("\nSampling training data...")


if len(train_full) > TRAIN_SAMPLE:

    indices = RNG.choice(
        len(train_full),
        size=TRAIN_SAMPLE,
        replace=False
    )

    train = train_full.iloc[
        indices
    ].copy()

else:

    train = train_full.copy()


del train_full

gc.collect()


print(
    f"Training rows: "
    f"{len(train):,}"
)

print(
    f"Validation rows: "
    f"{len(valid):,}"
)


# --------------------------------------------------
# RUN EACH HORIZON
# --------------------------------------------------

results = []


for horizon in HORIZONS:

    print("\n" + "=" * 70)

    print(
        f"HORIZON: {horizon} MINUTES"
    )

    print("=" * 70)


    target = f"target_{horizon}m"


    # Remove rows where the future target
    # could not be calculated.

    train_h = train.dropna(
        subset=[target]
    )

    valid_h = valid.dropna(
        subset=[target]
    )


    X_train = train_h[
        FEATURES
    ].astype("float32")

    y_train = train_h[
        target
    ].astype("float32")


    X_valid = valid_h[
        FEATURES
    ].astype("float32")

    y_valid = valid_h[
        target
    ].astype("float32")


    print(
        f"Train:      {len(y_train):,}"
    )

    print(
        f"Validation: {len(y_valid):,}"
    )


    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=300,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=0.1,

        random_state=42,

        n_jobs=2,
    )


    model.fit(

        X_train,

        y_train,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=30,
                verbose=False
            )
        ]
    )


    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(
        X_valid
    )

    actual = y_valid.to_numpy()


    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    mse = np.mean(
        (predictions - actual) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(
            predictions - actual
        )
    )

    correlation = np.corrcoef(
        predictions,
        actual
    )[0, 1]

    direction = np.mean(
        (predictions > 0)
        ==
        (actual > 0)
    )

    zero_mse = np.mean(
        actual ** 2
    )

    improvement = (
        1 -
        mse / zero_mse
    ) * 100


    # --------------------------------------------------
    # PRINT
    # --------------------------------------------------

    print(
        f"Best iteration: "
        f"{model.best_iteration_}"
    )

    print(
        f"MSE:             "
        f"{mse:.10e}"
    )

    print(
        f"Zero MSE:        "
        f"{zero_mse:.10e}"
    )

    print(
        f"MSE improvement: "
        f"{improvement:.3f}%"
    )

    print(
        f"RMSE:            "
        f"{rmse:.10e}"
    )

    print(
        f"MAE:             "
        f"{mae:.10e}"
    )

    print(
        f"Correlation:     "
        f"{correlation:.6f}"
    )

    print(
        f"Direction:       "
        f"{direction * 100:.2f}%"
    )


    results.append({

        "horizon": horizon,

        "best_iteration":
            model.best_iteration_,

        "mse": mse,

        "zero_mse":
            zero_mse,

        "improvement_percent":
            improvement,

        "rmse": rmse,

        "mae": mae,

        "correlation":
            correlation,

        "direction_accuracy":
            direction,
    })


    del model
    del train_h
    del valid_h
    del X_train
    del y_train
    del X_valid
    del y_valid
    del predictions

    gc.collect()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

result_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("HORIZON SUMMARY")
print("=" * 70)


print(
    result_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print("\nDone.")
