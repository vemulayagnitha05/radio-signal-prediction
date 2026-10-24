import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


DATA_PATH = "data/scanner_data_lte.parquet"

BLOCK_SIZE = 0.002
TEST_FRACTION = 0.20
SEEDS = [1, 21, 42, 84, 123]

FEATURES = [
    "latitude",
    "longitude",
    "frequency_khz",
    "channel_number",
    "hour",
    "day_of_week",
    "month",
    "minute_of_day_sin",
    "minute_of_day_cos",
    "week_sin",
    "week_cos",
    "time_period",
]

TARGET = "rsrp_dbm"


def make_features(df):
    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_of_week
    df["month"] = df["time"].dt.month

    minute_of_day = (
        df["time"].dt.hour * 60
        + df["time"].dt.minute
        + df["time"].dt.second / 60.0
    )

    df["minute_of_day_sin"] = np.sin(
        2 * np.pi * minute_of_day / 1440.0
    )
    df["minute_of_day_cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440.0
    )

    week_number = df["time"].dt.isocalendar().week.astype(int)

    df["week_sin"] = np.sin(
        2 * np.pi * week_number / 52.0
    )
    df["week_cos"] = np.cos(
        2 * np.pi * week_number / 52.0
    )

    df["time_period"] = (
        df["time"].dt.hour // 6
    ).astype(int)

    return df


def make_spatial_split(df, seed):
    block_lat = np.floor(
        df["latitude"] / BLOCK_SIZE
    ).astype(int)

    block_lon = np.floor(
        df["longitude"] / BLOCK_SIZE
    ).astype(int)

    blocks = (
        block_lat.astype(str)
        + "_"
        + block_lon.astype(str)
    )

    unique_blocks = blocks.unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)

    n_test = int(
        len(unique_blocks) * TEST_FRACTION
    )

    test_blocks = set(unique_blocks[:n_test])

    test_mask = blocks.isin(test_blocks)

    return ~test_mask, test_mask


def evaluate_model(X_train, y_train, X_test, y_test, loss):
    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(
        y_test,
        predictions,
    ))

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return model, predictions, rmse, mae, r2


print("Loading dataset...")

df = pd.read_parquet(DATA_PATH)

print(f"Rows: {len(df):,}")

df = make_features(df)

print("Features created.")
print(f"Feature count: {len(FEATURES)}")

results = []

for seed in SEEDS:

    print()
    print("=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    train_mask, test_mask = make_spatial_split(
        df,
        seed,
    )

    train = df.loc[train_mask]
    test = df.loc[test_mask]

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(
        f"Train rows: {len(train):,} | "
        f"Test rows: {len(test):,}"
    )

    # ---------------------------------------------------------
    # Model 1: squared-error loss
    # ---------------------------------------------------------

    print("\nTraining squared-error model...")

    (
        model_sq,
        pred_sq,
        rmse_sq,
        mae_sq,
        r2_sq,
    ) = evaluate_model(
        X_train,
        y_train,
        X_test,
        y_test,
        "squared_error",
    )

    print(
        f"Squared error -> "
        f"RMSE: {rmse_sq:.3f} dB | "
        f"MAE: {mae_sq:.3f} dB | "
        f"R²: {r2_sq:.3f}"
    )

    # ---------------------------------------------------------
    # Model 2: absolute-error loss
    # ---------------------------------------------------------

    print("\nTraining absolute-error model...")

    (
        model_abs,
        pred_abs,
        rmse_abs,
        mae_abs,
        r2_abs,
    ) = evaluate_model(
        X_train,
        y_train,
        X_test,
        y_test,
        "absolute_error",
    )

    print(
        f"Absolute error -> "
        f"RMSE: {rmse_abs:.3f} dB | "
        f"MAE: {mae_abs:.3f} dB | "
        f"R²: {r2_abs:.3f}"
    )

    # ---------------------------------------------------------
    # Tail error analysis
    # ---------------------------------------------------------

    weak_mask = y_test < -110
    strong_mask = y_test > -70

    weak_mae_sq = np.nan
    weak_mae_abs = np.nan
    strong_mae_sq = np.nan
    strong_mae_abs = np.nan

    if weak_mask.sum() > 0:
        weak_mae_sq = mean_absolute_error(
            y_test[weak_mask],
            pred_sq[weak_mask],
        )

        weak_mae_abs = mean_absolute_error(
            y_test[weak_mask],
            pred_abs[weak_mask],
        )

    if strong_mask.sum() > 0:
        strong_mae_sq = mean_absolute_error(
            y_test[strong_mask],
            pred_sq[strong_mask],
        )

        strong_mae_abs = mean_absolute_error(
            y_test[strong_mask],
            pred_abs[strong_mask],
        )

    results.append(
        {
            "seed": seed,

            "squared_rmse_db": rmse_sq,
            "squared_mae_db": mae_sq,
            "squared_r2": r2_sq,

            "absolute_rmse_db": rmse_abs,
            "absolute_mae_db": mae_abs,
            "absolute_r2": r2_abs,

            "weak_count": int(weak_mask.sum()),
            "weak_mae_squared_db": weak_mae_sq,
            "weak_mae_absolute_db": weak_mae_abs,

            "strong_count": int(strong_mask.sum()),
            "strong_mae_squared_db": strong_mae_sq,
            "strong_mae_absolute_db": strong_mae_abs,
        }
    )


results_df = pd.DataFrame(results)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

metrics = [
    (
        "Squared-error RMSE",
        "squared_rmse_db",
    ),
    (
        "Absolute-error RMSE",
        "absolute_rmse_db",
    ),
    (
        "Squared-error MAE",
        "squared_mae_db",
    ),
    (
        "Absolute-error MAE",
        "absolute_mae_db",
    ),
    (
        "Squared-error R²",
        "squared_r2",
    ),
    (
        "Absolute-error R²",
        "absolute_r2",
    ),
]

for label, column in metrics:
    mean_value = results_df[column].mean()
    std_value = results_df[column].std()

    print(
        f"{label}: "
        f"{mean_value:.3f} ± {std_value:.3f}"
    )

print()
print("Tail MAE:")

print(
    "Weak signal (< -110 dBm):"
)
print(
    f"  Squared error: "
    f"{results_df['weak_mae_squared_db'].mean():.3f} dB"
)
print(
    f"  Absolute error: "
    f"{results_df['weak_mae_absolute_db'].mean():.3f} dB"
)

print(
    "Strong signal (> -70 dBm):"
)
print(
    f"  Squared error: "
    f"{results_df['strong_mae_squared_db'].mean():.3f} dB"
)
print(
    f"  Absolute error: "
    f"{results_df['strong_mae_absolute_db'].mean():.3f} dB"
)

output_path = "results/target_loss_comparison.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print()
print(f"Saved results to: {output_path}")
