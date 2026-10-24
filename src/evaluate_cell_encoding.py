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


def add_time_features(df):

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


def add_cell_training_feature(
    train_df,
    test_df,
):

    train_df = train_df.copy()
    test_df = test_df.copy()

    training_mean = train_df[
        "rsrp_dbm"
    ].mean()

    cell_mean = (
        train_df
        .groupby("cell_id")["rsrp_dbm"]
        .mean()
    )

    train_df["cell_mean_rsrp"] = (
        train_df["cell_id"]
        .map(cell_mean)
    )

    test_df["cell_mean_rsrp"] = (
        test_df["cell_id"]
        .map(cell_mean)
        .fillna(training_mean)
    )

    unseen_cells = (
        test_df["cell_mean_rsrp"]
        == training_mean
    )

    return (
        train_df,
        test_df,
        int(unseen_cells.sum()),
        len(cell_mean),
    )


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
    print("LEAKAGE-SAFE CELL ENCODING EXPERIMENT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    df = add_time_features(df)

    base_features = [
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

    cell_features = base_features + [
        "cell_mean_rsrp",
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
        # Model 1: spatial + network context
        # ----------------------------------------------------

        baseline_rmse, baseline_mae, baseline_r2 = (
            evaluate_model(
                train_df,
                test_df,
                base_features,
            )
        )

        print("\nModel 1: Spatial + Network")
        print(
            f"RMSE: {baseline_rmse:.3f} dB"
        )
        print(
            f"MAE:  {baseline_mae:.3f} dB"
        )
        print(
            f"R²:   {baseline_r2:.3f}"
        )

        # ----------------------------------------------------
        # Model 2: add leakage-safe cell mean
        # ----------------------------------------------------

        (
            train_cell,
            test_cell,
            unseen_cells,
            training_cells,
        ) = add_cell_training_feature(
            train_df,
            test_df,
        )

        cell_rmse, cell_mae, cell_r2 = (
            evaluate_model(
                train_cell,
                test_cell,
                cell_features,
            )
        )

        print("\nModel 2: + Training Cell Mean")
        print(
            f"Training cells: {training_cells:,}"
        )
        print(
            f"Unseen test cells: {unseen_cells:,}"
        )
        print(
            f"RMSE: {cell_rmse:.3f} dB"
        )
        print(
            f"MAE:  {cell_mae:.3f} dB"
        )
        print(
            f"R²:   {cell_r2:.3f}"
        )

        results.append({
            "seed": seed,
            "model_1_rmse_db": baseline_rmse,
            "model_1_mae_db": baseline_mae,
            "model_1_r2": baseline_r2,
            "model_2_rmse_db": cell_rmse,
            "model_2_mae_db": cell_mae,
            "model_2_r2": cell_r2,
            "training_cells": training_cells,
            "unseen_test_cells": unseen_cells,
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("CELL ENCODING SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "seed",
                "model_1_rmse_db",
                "model_2_rmse_db",
                "model_1_r2",
                "model_2_r2",
                "unseen_test_cells",
            ]
        ].to_string(index=False)
    )

    print("\nMean ± Std")
    print("-" * 50)

    print(
        f"Model 1 RMSE: "
        f"{results_df['model_1_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['model_1_rmse_db'].std():.3f} dB"
    )

    print(
        f"Model 2 RMSE: "
        f"{results_df['model_2_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['model_2_rmse_db'].std():.3f} dB"
    )

    print(
        f"Model 1 R²:   "
        f"{results_df['model_1_r2'].mean():.3f} "
        f"± "
        f"{results_df['model_1_r2'].std():.3f}"
    )

    print(
        f"Model 2 R²:   "
        f"{results_df['model_2_r2'].mean():.3f} "
        f"± "
        f"{results_df['model_2_r2'].std():.3f}"
    )

    output_path = (
        "results/cell_encoding.csv"
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
