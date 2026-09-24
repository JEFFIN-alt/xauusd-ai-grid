import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FILE = (
    "data/processed/"
    "xauusd_m1_ml_features_v3.csv"
)

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

HORIZONS = [5, 15, 30, 60]

# Deterministic systematic sampling.
# Approximately 1/5 of the training rows.
TRAIN_STRIDE = 5

TARGET_PREFIX = "future_return"

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

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
    f"{TARGET_PREFIX}_{h}m"
    for h in HORIZONS
]


USECOLS = [
    "timestamp"
] + FEATURES + TARGETS


DTYPES = {
    "timestamp": "int64"
}

for column in FEATURES + TARGETS:
    DTYPES[column] = "float32"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("V3 MULTI-HORIZON LIGHTGBM BASELINE")
print("=" * 70)

print(
    f"\nTraining:   "
    f"{TRAIN_START} -> {TRAIN_END}"
)

print(
    f"Validation: "
    f"{VALID_START} -> {VALID_END}"
)

print(
    f"Test:       NOT USED"
)

print(
    f"\nTraining sampling: every {TRAIN_STRIDE}th row"
)


# ============================================================
# LOAD DATA
# ============================================================

train_parts = []

valid_parts = []

global_row = 0


print("\nReading V3 in chunks...")


for chunk_number, chunk in enumerate(

    pd.read_csv(
        FILE,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
    ),

    start=1,
):

    row_index = np.arange(
        global_row,
        global_row + len(chunk)
    )

    global_row += len(chunk)


    dt = pd.to_datetime(
        chunk["timestamp"],
        unit="ms",
        utc=True
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_mask = (
        (dt >= TRAIN_START)
        &
        (dt <= TRAIN_END)
    )


    if train_mask.any():

        train_chunk = chunk.loc[
            train_mask
        ].copy()


        # Systematic deterministic sampling.
        sampled = (
            row_index[
                train_mask.to_numpy()
            ]
            % TRAIN_STRIDE
            == 0
        )


        train_chunk = train_chunk.loc[
            sampled
        ]


        if not train_chunk.empty:

            train_parts.append(
                train_chunk
            )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid_mask = (
        (dt >= VALID_START)
        &
        (dt <= VALID_END)
    )


    if valid_mask.any():

        valid_parts.append(
            chunk.loc[
                valid_mask
            ].copy()
        )


    del chunk
    del dt
    del row_index

    if chunk_number % 10 == 0:

        current_train = sum(
            len(x)
            for x in train_parts
        )

        current_valid = sum(
            len(x)
            for x in valid_parts
        )

        print(
            f"Chunks: {chunk_number:02d} | "
            f"Train sample: {current_train:,} | "
            f"Validation: {current_valid:,}"
        )

    gc.collect()


# ============================================================
# COMBINE
# ============================================================

train = pd.concat(
    train_parts,
    ignore_index=True
)

valid = pd.concat(
    valid_parts,
    ignore_index=True
)

del train_parts
del valid_parts

gc.collect()


print(
    f"\nTraining sample rows: "
    f"{len(train):,}"
)

print(
    f"Validation rows: "
    f"{len(valid):,}"
)


# ============================================================
# TRAIN EACH HORIZON
# ============================================================

results = []


for horizon in HORIZONS:

    target = (
        f"{TARGET_PREFIX}_{horizon}m"
    )


    print("\n" + "=" * 70)

    print(
        f"HORIZON: {horizon} MINUTES"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Only use rows with a valid target for this horizon.
    # --------------------------------------------------------

    train_h = train.loc[
        train[target].notna()
    ]

    valid_h = valid.loc[
        valid[target].notna()
    ]


    X_train = train_h[
        FEATURES
    ]

    y_train = train_h[
        target
    ]


    X_valid = valid_h[
        FEATURES
    ]

    y_valid = valid_h[
        target
    ]


    print(
        f"Train target rows: "
        f"{len(y_train):,}"
    )

    print(
        f"Validation target rows: "
        f"{len(y_valid):,}"
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = lgb.LGBMRegressor(

        objective="regression",

        n_estimators=500,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        colsample_bytree=0.8,

        reg_alpha=0.1,

        reg_lambda=0.1,

        random_state=RANDOM_STATE,

        n_jobs=2,

        verbosity=-1,
    )


    print("\nTraining...")


    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid
            )
        ],

        eval_names=[
            "validation"
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=40,
                verbose=False
            )
        ]
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = model.predict(
        X_valid
    )

    actual = (
        y_valid
        .to_numpy(dtype=np.float64)
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    mse = np.mean(
        (predictions - actual) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(
            predictions - actual
        )
    )


    # --------------------------------------------------------
    # ZERO BASELINE
    # --------------------------------------------------------

    zero_mse = np.mean(
        actual ** 2
    )

    zero_rmse = np.sqrt(
        zero_mse
    )

    zero_mae = np.mean(
        np.abs(actual)
    )


    improvement = (
        1.0 -
        mse / zero_mse
    ) * 100.0


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    if (
        np.std(predictions) > 0
        and np.std(actual) > 0
    ):

        correlation = np.corrcoef(
            predictions,
            actual
        )[0, 1]

    else:

        correlation = np.nan


    # --------------------------------------------------------
    # DIRECTIONAL ACCURACY
    # --------------------------------------------------------

    direction_accuracy = np.mean(
        (predictions > 0)
        ==
        (actual > 0)
    )


    # --------------------------------------------------------
    # DIRECTIONAL ACCURACY
    # excluding exact-zero actual returns
    # --------------------------------------------------------

    nonzero_actual = (
        actual != 0
    )


    if nonzero_actual.any():

        nonzero_direction_accuracy = np.mean(

            (
                predictions[
                    nonzero_actual
                ]
                > 0
            )
            ==
            (
                actual[
                    nonzero_actual
                ]
                > 0
            )
        )

    else:

        nonzero_direction_accuracy = np.nan


    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    prediction_mean = np.mean(
        predictions
    )

    prediction_std = np.std(
        predictions
    )

    prediction_min = np.min(
        predictions
    )

    prediction_max = np.max(
        predictions
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nBest iteration: "
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
        f"{improvement:.4f}%"
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
        f"Zero MAE:        "
        f"{zero_mae:.10e}"
    )

    print(
        f"Correlation:     "
        f"{correlation:.6f}"
    )

    print(
        f"Direction:       "
        f"{direction_accuracy * 100:.2f}%"
    )

    print(
        f"Direction "
        f"(nonzero actual): "
        f"{nonzero_direction_accuracy * 100:.2f}%"
    )

    print(
        f"Prediction mean: "
        f"{prediction_mean:.10e}"
    )

    print(
        f"Prediction std:  "
        f"{prediction_std:.10e}"
    )

    print(
        f"Prediction min:  "
        f"{prediction_min:.10e}"
    )

    print(
        f"Prediction max:  "
        f"{prediction_max:.10e}"
    )


    results.append({

        "horizon": horizon,

        "train_rows": len(y_train),

        "validation_rows": len(y_valid),

        "best_iteration":
            model.best_iteration_,

        "mse": mse,

        "zero_mse":
            zero_mse,

        "mse_improvement_percent":
            improvement,

        "rmse":
            rmse,

        "mae":
            mae,

        "zero_mae":
            zero_mae,

        "correlation":
            correlation,

        "direction_accuracy":
            direction_accuracy,

        "nonzero_direction_accuracy":
            nonzero_direction_accuracy,

        "prediction_mean":
            prediction_mean,

        "prediction_std":
            prediction_std,

        "prediction_min":
            prediction_min,

        "prediction_max":
            prediction_max,
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


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("V3 HORIZON BASELINE SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
