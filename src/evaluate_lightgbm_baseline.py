import pandas as pd
import numpy as np
import lightgbm as lgb
import gc

FILE = "data/processed/xauusd_m1_15m_features.csv"
MODEL_FILE = "models/lightgbm_baseline/lightgbm_baseline.txt"

CHUNK_SIZE = 100_000

VALID_START = pd.Timestamp("2025-01-01", tz="UTC")
VALID_END = pd.Timestamp("2025-12-31 23:45:00", tz="UTC")

TARGET = "future_return_15m"


print("=" * 70)
print("LIGHTGBM BASELINE EVALUATION")
print("=" * 70)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading LightGBM model...")

model = lgb.Booster(
    model_file=MODEL_FILE
)

print("Model loaded.")


# --------------------------------------------------
# GET FEATURES
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
    TARGET,
}

FEATURES = [
    c for c in columns
    if c not in excluded
]

print(f"\nFeatures: {len(FEATURES)}")


# --------------------------------------------------
# STORAGE FOR RESULTS
# --------------------------------------------------

actual_all = []
predicted_all = []

valid_rows = 0


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

print("\nLoading validation data...")

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

    mask = (
        (dt >= VALID_START) &
        (dt <= VALID_END)
    )

    if mask.any():

        data = chunk.loc[
            mask,
            FEATURES + [TARGET]
        ].copy()

        # LightGBM prediction
        X = data[FEATURES].astype("float32")

        y = data[TARGET].astype("float32")

        predictions = model.predict(X)

        actual_all.append(
            y.to_numpy()
        )

        predicted_all.append(
            predictions
        )

        valid_rows += len(data)

        del data
        del X
        del y
        del predictions

    del chunk
    del dt

    if chunk_number % 10 == 0:
        print(
            f"Processed chunks: {chunk_number} | "
            f"validation rows: {valid_rows:,}"
        )

    gc.collect()


# --------------------------------------------------
# COMBINE
# --------------------------------------------------

actual = np.concatenate(actual_all)
predicted = np.concatenate(predicted_all)

del actual_all
del predicted_all

gc.collect()


print("\nValidation rows:", len(actual))


# --------------------------------------------------
# METRICS
# --------------------------------------------------

errors = predicted - actual

mse = np.mean(errors ** 2)

rmse = np.sqrt(mse)

mae = np.mean(np.abs(errors))


# --------------------------------------------------
# ZERO BASELINE
# --------------------------------------------------

zero_predictions = np.zeros_like(actual)

zero_errors = zero_predictions - actual

zero_mse = np.mean(
    zero_errors ** 2
)

zero_rmse = np.sqrt(zero_mse)

zero_mae = np.mean(
    np.abs(zero_errors)
)


# --------------------------------------------------
# MEAN BASELINE
# --------------------------------------------------

mean_prediction = np.mean(actual)

mean_predictions = np.full_like(
    actual,
    mean_prediction
)

mean_errors = (
    mean_predictions - actual
)

mean_mse = np.mean(
    mean_errors ** 2
)

mean_rmse = np.sqrt(mean_mse)

mean_mae = np.mean(
    np.abs(mean_errors)
)


# --------------------------------------------------
# CORRELATION
# --------------------------------------------------

pearson = np.corrcoef(
    predicted,
    actual
)[0, 1]


# --------------------------------------------------
# DIRECTIONAL ACCURACY
# --------------------------------------------------

actual_direction = actual > 0
predicted_direction = predicted > 0

directional_accuracy = np.mean(
    actual_direction ==
    predicted_direction
)


# --------------------------------------------------
# SIGNAL-ONLY DIRECTIONAL ACCURACY
# --------------------------------------------------

nonzero_prediction = predicted != 0

if nonzero_prediction.any():

    signal_direction_accuracy = np.mean(
        predicted_direction[nonzero_prediction]
        ==
        actual_direction[nonzero_prediction]
    )

else:

    signal_direction_accuracy = np.nan


# --------------------------------------------------
# PREDICTION STATISTICS
# --------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nMODEL")

print(
    f"MSE:              {mse:.10e}"
)

print(
    f"RMSE:             {rmse:.10e}"
)

print(
    f"MAE:              {mae:.10e}"
)

print(
    f"Correlation:      {pearson:.6f}"
)

print(
    f"Direction accuracy: "
    f"{directional_accuracy * 100:.2f}%"
)

print(
    f"Signal direction accuracy: "
    f"{signal_direction_accuracy * 100:.2f}%"
)


print("\nZERO-RETURN BASELINE")

print(
    f"MSE:              {zero_mse:.10e}"
)

print(
    f"RMSE:             {zero_rmse:.10e}"
)

print(
    f"MAE:              {zero_mae:.10e}"
)


print("\nMEAN-RETURN BASELINE")

print(
    f"Mean prediction:  {mean_prediction:.10e}"
)

print(
    f"MSE:              {mean_mse:.10e}"
)

print(
    f"RMSE:             {mean_rmse:.10e}"
)

print(
    f"MAE:              {mean_mae:.10e}"
)


# --------------------------------------------------
# IMPROVEMENT OVER ZERO BASELINE
# --------------------------------------------------

mse_improvement = (
    1 -
    (mse / zero_mse)
) * 100

rmse_improvement = (
    1 -
    (rmse / zero_rmse)
) * 100


print("\nMODEL IMPROVEMENT OVER ZERO BASELINE")

print(
    f"MSE improvement:  "
    f"{mse_improvement:.2f}%"
)

print(
    f"RMSE improvement: "
    f"{rmse_improvement:.2f}%"
)


# --------------------------------------------------
# PREDICTION DISTRIBUTION
# --------------------------------------------------

print("\nPREDICTION DISTRIBUTION")

print(
    f"Mean:   {np.mean(predicted):.10e}"
)

print(
    f"Std:    {np.std(predicted):.10e}"
)

print(
    f"Min:    {np.min(predicted):.10e}"
)

print(
    f"P25:    {np.percentile(predicted, 25):.10e}"
)

print(
    f"Median: {np.median(predicted):.10e}"
)

print(
    f"P75:    {np.percentile(predicted, 75):.10e}"
)

print(
    f"Max:    {np.max(predicted):.10e}"
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
