import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc

FILE = "data/processed/xauusd_m1_15m_features.csv"

CHUNK_SIZE = 100_000
TRAIN_SAMPLE_SIZE = 600_000

TRAIN_START = pd.Timestamp("2019-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2024-12-31 23:45:00", tz="UTC")

VALID_START = pd.Timestamp("2025-01-01", tz="UTC")
VALID_END = pd.Timestamp("2025-12-31 23:45:00", tz="UTC")

TARGET = "future_return_15m"

OUTPUT_DIR = "models/lightgbm_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("LIGHTGBM BASELINE")
print("=" * 70)


# --------------------------------------------------
# READ COLUMN NAMES
# --------------------------------------------------

print("\nReading column names...")

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
    TARGET,
}

FEATURES = [
    c for c in columns
    if c not in excluded
]

print(f"Total columns: {len(columns)}")
print(f"Features:      {len(FEATURES)}")
print(f"Target:        {TARGET}")


# --------------------------------------------------
# RESERVOIRS FOR TIME-DISTRIBUTED TRAINING SAMPLE
# --------------------------------------------------

print("\nCollecting training sample...")

rng = np.random.default_rng(42)

train_sample_parts = []
valid_parts = []

train_rows_seen = 0


for chunk_number, chunk in enumerate(
    pd.read_csv(
        FILE,
        usecols=["timestamp"] + FEATURES + [TARGET],
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    dt = pd.to_datetime(
        chunk["timestamp"],
        unit="ms",
        utc=True,
    )

    # -----------------------------
    # TRAIN
    # -----------------------------

    train_mask = (
        (dt >= TRAIN_START) &
        (dt <= TRAIN_END)
    )

    train_chunk = chunk.loc[
        train_mask,
        FEATURES + [TARGET]
    ]

    if len(train_chunk) > 0:

        train_rows_seen += len(train_chunk)

        # Fraction needed to approach 600k total
        remaining = TRAIN_SAMPLE_SIZE - sum(
            len(x) for x in train_sample_parts
        )

        if remaining > 0:

            take = min(
                len(train_chunk),
                max(
                    1,
                    int(
                        TRAIN_SAMPLE_SIZE
                        * len(train_chunk)
                        / 2_619_865
                    )
                )
            )

            if take > len(train_chunk):
                take = len(train_chunk)

            indices = rng.choice(
                len(train_chunk),
                size=take,
                replace=False,
            )

            train_sample_parts.append(
                train_chunk.iloc[indices]
            )

    # -----------------------------
    # VALIDATION
    # -----------------------------

    valid_mask = (
        (dt >= VALID_START) &
        (dt <= VALID_END)
    )

    valid_chunk = chunk.loc[
        valid_mask,
        FEATURES + [TARGET]
    ]

    if len(valid_chunk) > 0:
        valid_parts.append(valid_chunk)

    del chunk
    del dt

    if chunk_number % 10 == 0:
        current = sum(
            len(x) for x in train_sample_parts
        )

        print(
            f"Processed chunks: {chunk_number} | "
            f"training sample: {current:,}"
        )

    gc.collect()


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

print("\nCombining datasets...")

train_df = pd.concat(
    train_sample_parts,
    ignore_index=True,
)

valid_df = pd.concat(
    valid_parts,
    ignore_index=True,
)

del train_sample_parts
del valid_parts

gc.collect()


# --------------------------------------------------
# CONVERT TO FLOAT32
# --------------------------------------------------

print("\nConverting numerical data to float32...")

train_df = train_df.astype("float32")
valid_df = valid_df.astype("float32")


print(f"\nTraining rows:   {len(train_df):,}")
print(f"Validation rows: {len(valid_df):,}")

print(
    f"Training memory: "
    f"{train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
)

print(
    f"Validation memory: "
    f"{valid_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
)


# --------------------------------------------------
# SEPARATE FEATURES / TARGET
# --------------------------------------------------

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_valid = valid_df[FEATURES]
y_valid = valid_df[TARGET]


print("\nX_train:", X_train.shape)
print("X_valid:", X_valid.shape)


# --------------------------------------------------
# LIGHTGBM MODEL
# --------------------------------------------------

print("\nCreating LightGBM model...")

model = lgb.LGBMRegressor(

    objective="regression",

    n_estimators=1000,

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


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting training...")
print("=" * 70)

model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_valid, y_valid),
    ],

    eval_names=[
        "train",
        "validation",
    ],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50
        ),

        lgb.log_evaluation(
            period=25
        ),
    ],
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

model_path = (
    f"{OUTPUT_DIR}/"
    "lightgbm_baseline.txt"
)

model.booster_.save_model(
    model_path
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest iteration: "
    f"{model.best_iteration_}"
)

print(
    f"Model saved to:\n"
    f"{model_path}"
)

print("\nDone.")
