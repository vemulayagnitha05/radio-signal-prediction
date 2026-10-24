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

    test_blocks = set(
        blocks[:n_test]
    )

    test_mask = df[
        "spatial_block"
    ].isin(test_blocks)

    train_df = df.loc[
        ~test_mask
    ].copy()

    test_df = df.loc[
        test_mask
    ].copy()

    return train_df, test_df


def evaluate_predictions(
    y_true,
    predictions,
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions,
        )
    )

    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    r2 = r2_score(
        y_true,
        predictions,
    )

    return rmse, mae, r2


def global_mean_baseline(
    train_df,
    test_df,
):

    training_mean = train_df[
        "rsrp_dbm"
    ].mean()

    predictions = np.full(
        len(test_df),
        training_mean,
    )

    return evaluate_predictions(
        test_df["rsrp_dbm"],
        predictions,
    )


def operator_channel_baseline(
    train_df,
    test_df,
):

    training_mean = train_df[
        "rsrp_dbm"
    ].mean()

    group_means = (
        train_df
        .groupby(
            [
                "operator",
                "channel_number",
            ]
        )["rsrp_dbm"]
        .mean()
    )

    test_groups = pd.MultiIndex.from_frame(
        test_df[
            [
                "operator",
                "channel_number",
            ]
        ]
    )

    predictions = (
        group_means
        .reindex(test_groups)
        .fillna(training_mean)
        .to_numpy()
    )

    return evaluate_predictions(
        test_df["rsrp_dbm"],
        predictions,
    )


def add_temporal_features(df):

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = (
        df["time"].dt.day_of_week
    )
    df["month"] = df["time"].dt.month

    minutes = (
        df["time"].dt.hour * 60
        + df["time"].dt.minute
        + df["time"].dt.second / 60
    )

    df["minute_of_day_sin"] = np.sin(
        2 * np.pi * minutes / 1440
    )

    df["minute_of_day_cos"] = np.cos(
        2 * np.pi * minutes / 1440
    )

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


def ml_model(
    train_df,
    test_df,
):

    features = [
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

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42,
    )

    model.fit(
        train_df[features],
        train_df["rsrp_dbm"],
    )

    predictions = model.predict(
        test_df[features]
    )

    return evaluate_predictions(
        test_df["rsrp_dbm"],
        predictions,
    )


def main():

    print("=" * 70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    df = add_temporal_features(df)

    results = []

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"SEED: {seed}")
        print("-" * 70)

        train_df, test_df = (
            make_spatial_split(
                df,
                seed,
            )
        )

        print(
            f"Train rows: {len(train_df):,}"
        )

        print(
            f"Test rows:  {len(test_df):,}"
        )

        # ----------------------------------------------------
        # Global mean
        # ----------------------------------------------------

        global_rmse, global_mae, global_r2 = (
            global_mean_baseline(
                train_df,
                test_df,
            )
        )

        print("\nGlobal Mean Baseline")

        print(
            f"RMSE: {global_rmse:.3f} dB"
        )

        print(
            f"MAE:  {global_mae:.3f} dB"
        )

        print(
            f"R²:   {global_r2:.3f}"
        )

        # ----------------------------------------------------
        # Operator + channel
        # ----------------------------------------------------

        group_rmse, group_mae, group_r2 = (
            operator_channel_baseline(
                train_df,
                test_df,
            )
        )

        print("\nOperator + Channel Mean")

        print(
            f"RMSE: {group_rmse:.3f} dB"
        )

        print(
            f"MAE:  {group_mae:.3f} dB"
        )

        print(
            f"R²:   {group_r2:.3f}"
        )

        # ----------------------------------------------------
        # ML model
        # ----------------------------------------------------

        ml_rmse, ml_mae, ml_r2 = ml_model(
            train_df,
            test_df,
        )

        print("\nSpatial + Rich Temporal ML")

        print(
            f"RMSE: {ml_rmse:.3f} dB"
        )

        print(
            f"MAE:  {ml_mae:.3f} dB"
        )

        print(
            f"R²:   {ml_r2:.3f}"
        )

        results.append({
            "seed": seed,
            "global_rmse_db": global_rmse,
            "global_mae_db": global_mae,
            "global_r2": global_r2,
            "group_rmse_db": group_rmse,
            "group_mae_db": group_mae,
            "group_r2": group_r2,
            "ml_rmse_db": ml_rmse,
            "ml_mae_db": ml_mae,
            "ml_r2": ml_r2,
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "seed",
                "global_rmse_db",
                "group_rmse_db",
                "ml_rmse_db",
                "global_r2",
                "group_r2",
                "ml_r2",
            ]
        ].to_string(index=False)
    )

    print("\nMean ± Std")
    print("-" * 50)

    print(
        f"Global Mean RMSE: "
        f"{results_df['global_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['global_rmse_db'].std():.3f} dB"
    )

    print(
        f"Operator+Channel RMSE: "
        f"{results_df['group_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['group_rmse_db'].std():.3f} dB"
    )

    print(
        f"ML RMSE: "
        f"{results_df['ml_rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['ml_rmse_db'].std():.3f} dB"
    )

    print(
        f"\nGlobal Mean R²: "
        f"{results_df['global_r2'].mean():.3f} "
        f"± "
        f"{results_df['global_r2'].std():.3f}"
    )

    print(
        f"Operator+Channel R²: "
        f"{results_df['group_r2'].mean():.3f} "
        f"± "
        f"{results_df['group_r2'].std():.3f}"
    )

    print(
        f"ML R²: "
        f"{results_df['ml_r2'].mean():.3f} "
        f"± "
        f"{results_df['ml_r2'].std():.3f}"
    )

    output_path = (
        "results/baseline_comparison.csv"
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
