import gc

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

HORIZONS = [5, 15, 30, 60]

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
    f"future_return_{h}m"
    for h in HORIZONS
]

USECOLS = [
    "timestamp"
] + FEATURES + TARGETS


# ============================================================
# WALK-FORWARD FOLDS
# ============================================================

FOLDS = [
    (
        "2021-12-31 23:59:59",
        "2022-01-01",
        "2022-12-31 23:45:00",
    ),
    (
        "2022-12-31 23:59:59",
        "2023-01-01",
        "2023-12-31 23:45:00",
    ),
    (
        "2023-12-31 23:59:59",
        "2024-01-01",
        "2024-12-31 23:45:00",
    ),
    (
        "2024-12-31 23:59:59",
        "2025-01-01",
        "2025-12-31 23:45:00",
    ),
]


# Maximum lookahead used in this experiment.
MAX_HORIZON_MIN = 60

# Feature warmup.
FEATURE_WARMUP_MIN = 480

# Deterministic systematic sampling.
TRAIN_STRIDE = 5


print("=" * 70)
print("WALK-FORWARD XAUUSD HORIZON TEST")
print("=" * 70)

print("\n2026 TEST SET: NOT USED")

print(
    f"Training stride: every {TRAIN_STRIDE}th V3 row"
)


# ============================================================
# LOAD ALL FOLD DATA
#
# We keep only the years needed for folds.
# One fold is trained/evaluated at a time.
# ============================================================

results = []


for fold_number, (
    train_end_string,
    valid_start_string,
    valid_end_string,
) in enumerate(FOLDS, start=1):

    print("\n" + "=" * 70)

    print(f"FOLD {fold_number}")

    print("=" * 70)


    train_end = pd.Timestamp(
        train_end_string,
        tz="UTC"
    )

    valid_start = pd.Timestamp(
        valid_start_string,
        tz="UTC"
    )

    valid_end = pd.Timestamp(
        valid_end_string,
        tz="UTC"
    )


    # We use the full validation period from the start
    # of the validation year.
    #
    # The 480-minute feature warmup refers to the
    # beginning of validation, not an exclusion from
    # training.
    validation_feature_warmup_end = (
        valid_start +
        pd.Timedelta(
            minutes=FEATURE_WARMUP_MIN
        )
    )


    # --------------------------------------------------------
    # TRAINING DATA
    #
    # Purge max 60 minutes from train/validation boundary.
    # --------------------------------------------------------

    train_cutoff = (
        train_end -
        pd.Timedelta(
            minutes=MAX_HORIZON_MIN
        )
    )


    train_start = pd.Timestamp(
        "2019-01-01",
        tz="UTC"
    )


    print(
        f"\nTrain: "
        f"{train_start} -> {train_cutoff}"
    )

    print(
        f"Validation: "
        f"{validation_feature_warmup_end} -> {valid_end}"
    )


    train_parts = []

    valid_parts = []

    global_row = 0


    # --------------------------------------------------------
    # READ V3
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(

        pd.read_csv(
            FILE,
            usecols=USECOLS,
            dtype={
                "timestamp": "int64",
                **{
                    c: "float32"
                    for c in FEATURES + TARGETS
                },
            },
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


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_mask = (
            (dt >= train_start)
            &
            (dt <= train_cutoff)
        )


        if train_mask.any():

            train_chunk = chunk.loc[
                train_mask
            ].copy()


            selected_global_rows = (
                row_index[
                    train_mask.to_numpy()
                ]
            )


            sampled = (
                selected_global_rows
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


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        valid_mask = (
            (dt >= validation_feature_warmup_end)
            &
            (dt <= valid_end)
        )


        if valid_mask.any():

            valid_parts.append(
                chunk.loc[
                    valid_mask
                ].copy()
            )


        del row_index
        del chunk
        del dt

        if chunk_number % 10 == 0:

            print(
                f"Chunks: {chunk_number:02d}"
            )

        gc.collect()


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

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
        f"\nTrain rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(valid):,}"
    )


    # ========================================================
    # EACH HORIZON
    # ========================================================

    for horizon in HORIZONS:

        target = (
            f"future_return_{horizon}m"
        )


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


        print("\n" + "-" * 70)

        print(
            f"Fold {fold_number} | "
            f"Horizon {horizon}m"
        )

        print("-" * 70)


        model = lgb.LGBMRegressor(

            objective="regression",

            n_estimators=300,

            learning_rate=0.03,

            num_leaves=31,

            colsample_bytree=0.8,

            reg_alpha=0.1,

            reg_lambda=0.1,

            random_state=42,

            n_jobs=2,

            verbosity=-1,
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


        predictions = model.predict(
            X_valid
        )

        actual = (
            y_valid
            .to_numpy(
                dtype=np.float64
            )
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mse = np.mean(
            (predictions - actual) ** 2
        )

        zero_mse = np.mean(
            actual ** 2
        )

        improvement = (
            1 -
            mse / zero_mse
        ) * 100


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


        nonzero = (
            actual != 0
        )


        if nonzero.any():

            nonzero_direction = np.mean(

                (
                    predictions[nonzero]
                    > 0
                )
                ==
                (
                    actual[nonzero]
                    > 0
                )
            )

        else:

            nonzero_direction = np.nan


        print(
            f"Best iteration: "
            f"{model.best_iteration_}"
        )

        print(
            f"MSE improvement: "
            f"{improvement:.4f}%"
        )

        print(
            f"Correlation: "
            f"{correlation:.6f}"
        )

        print(
            f"Nonzero direction: "
            f"{nonzero_direction * 100:.2f}%"
        )


        results.append({

            "fold": fold_number,

            "validation_year":
                valid_start.year,

            "horizon": horizon,

            "train_rows":
                len(y_train),

            "validation_rows":
                len(y_valid),

            "best_iteration":
                model.best_iteration_,

            "mse":
                mse,

            "zero_mse":
                zero_mse,

            "mse_improvement_percent":
                improvement,

            "correlation":
                correlation,

            "nonzero_direction_accuracy":
                nonzero_direction,
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


    del train
    del valid

    gc.collect()


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)


print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("HORIZON AGGREGATES")
print("=" * 70)


summary = (
    results_df
    .groupby("horizon")
    .agg(
        folds=("fold", "count"),

        mean_improvement=(
            "mse_improvement_percent",
            "mean"
        ),

        median_improvement=(
            "mse_improvement_percent",
            "median"
        ),

        mean_correlation=(
            "correlation",
            "mean"
        ),

        median_correlation=(
            "correlation",
            "median"
        ),

        mean_nonzero_direction=(
            "nonzero_direction_accuracy",
            "mean"
        ),
    )
    .reset_index()
)


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
