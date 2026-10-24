import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/scanner_data_lte.parquet"

BLOCK_SIZE = 0.002
TEST_FRACTION = 0.20

SEEDS = [1, 21, 42, 84, 123]


def make_spatial_split(df, seed):

    df = df.copy()

    df["lat_block"] = np.floor(
        df["latitude"] / BLOCK_SIZE
    ).astype(int)

    df["lon_block"] = np.floor(
        df["longitude"] / BLOCK_SIZE
    ).astype(int)

    df["spatial_block"] = (
        df["lat_block"].astype(str)
        + "_"
        + df["lon_block"].astype(str)
    )

    blocks = np.array(
        df["spatial_block"].unique(),
        dtype=str,
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n_test = max(
        1,
        int(len(blocks) * TEST_FRACTION),
    )

    test_blocks = set(blocks[:n_test])

    test_mask = df["spatial_block"].isin(
        test_blocks
    )

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    return train_df, test_df


def add_basic_features(df):

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.day_of_week
    df["month"] = df["time"].dt.month

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    return df


def add_richer_temporal_features(df):

    df = df.copy()

    # Minutes since midnight
    minutes = (
        df["time"].dt.hour * 60
        + df["time"].dt.minute
        + df["time"].dt.second / 60
    )

    # Full daily cycle
    df["minute_of_day_sin"] = np.sin(
        2 * np.pi * minutes / 1440
    )

    df["minute_of_day_cos"] = np.cos(
        2 * np.pi * minutes / 1440
    )

    # Weekly cycle
    day_fraction = (
        df["time"].dt.day_of_week
        + minutes / 1440
    )

    df["week_sin"] = np.sin(
        2 * np.pi * day_fraction / 7
    )

    df["week_cos"] = np.cos(
        2 * np.pi * day_fraction / 7
    )

    # Coarse time-of-day category
    hour = df["time"].dt.hour

    df["time_period"] = pd.cut(
        hour,
        bins=[
            -1,
            6,
            12,
            18,
            24,
        ],
        labels=[
            0,
            1,
            2,
            3,
        ],
    ).astype(int)

    return df


def evaluate_model(
    train_df,
    test_df,
    features,
):

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )

    X_train = train_df[features]
    y_train = train_df["rsrp_dbm"]

    X_test = test_df[features]
    y_test = test_df["rsrp_dbm"]

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return rmse, mae, r2


def main():

    print("=" * 70)
    print("TEMPORAL FEATURE EXPERIMENT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    df = add_basic_features(df)
    df = add_richer_temporal_features(df)

    basic_features = [
        "latitude",
        "longitude",
        "frequency_khz",
        "channel_number",
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
    ]

    temporal_features = basic_features + [
        "minute_of_day_sin",
        "minute_of_day_cos",
        "week_sin",
        "week_cos",
        "time_period",
    ]

    results = []

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"SEED: {seed}")
        print("-" * 70)

        train_df, test_df = make_spatial_split(
            df,
            seed,
        )

        print(
            f"Train rows: {len(train_df):,}"
        )

        print(
            f"Test rows:  {len(test_df):,}"
        )

        # ----------------------------------------------------
        # Model 1: existing temporal features
        # ----------------------------------------------------

        basic_rmse, basic_mae, basic_r2 = (
            evaluate_model(
                train_df,
                test_df,
                basic_features,
            )
        )

        print("\nModel 1: Basic Temporal Features")

        print(
            f"RMSE: {basic_rmse:.3f} dB"
        )

        print(
            f"MAE:  {basic_mae:.3f} dB"
        )

        print(
            f"R²:   {basic_r2:.3f}"
        )

        # ----------------------------------------------------
        # Model 2: richer temporal features
        # ----------------------------------------------------

        temporal_rmse, temporal_mae, temporal_r2 = (
            evaluate_model(
                train_df,
                test_df,
                temporal_features,
            )
        )

        print("\nModel 2: Rich Temporal Features")

        print(
            f"RMSE: {temporal_rmse:.3f} dB"
        )

        print(
            f"MAE:  {temporal_mae:.3f} dB"
        )

        print(
            f"R²:   {temporal_r2:.3f}"
        )

        results.append({
            "seed": seed,
            "basic_rmse_db": basic_rmse,
            "basic_mae_db": basic_mae,
            "basic_r2": basic_r2,
            "temporal_rmse_db": temporal_rmse,
            "temporal_mae_db": temporal_mae,
            "temporal_r2": temporal_r2,
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("TEMPORAL FEATURE SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "seed",
                "basic_rmse_db",
                "temporal_rmse_db",
                "basic_r2",
                "temporal_r2",
            ]
        ].to_string(index=False)
    )

    print("\nMean ± Std")
    print("-" * 50)

    print(
        f"Basic RMSE: "
        f"{results_df['basic_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['basic_rmse_db'].std():.3f} dB"
    )

    print(
        f"Rich Temporal RMSE: "
        f"{results_df['temporal_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['temporal_rmse_db'].std():.3f} dB"
    )

    print(
        f"Basic R²: "
        f"{results_df['basic_r2'].mean():.3f} "
        f"± "
        f"{results_df['basic_r2'].std():.3f}"
    )

    print(
        f"Rich Temporal R²: "
        f"{results_df['temporal_r2'].mean():.3f} "
        f"± "
        f"{results_df['temporal_r2'].std():.3f}"
    )

    output_path = (
        "results/temporal_features.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
