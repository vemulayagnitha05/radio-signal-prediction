import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/scanner_data_lte.parquet"
OUTPUT_PATH = "results/spatial_context.csv"

BLOCK_SIZE = 0.002
TEST_FRACTION = 0.20
SEEDS = [1, 21, 42, 84, 123]


def create_features(df):
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
        2 * np.pi * minute_of_day / 1440
    )
    df["minute_of_day_cos"] = np.cos(
        2 * np.pi * minute_of_day / 1440
    )

    week_fraction = (
        df["day_of_week"] * 24
        + df["time"].dt.hour
    )

    df["week_sin"] = np.sin(
        2 * np.pi * week_fraction / (7 * 24)
    )
    df["week_cos"] = np.cos(
        2 * np.pi * week_fraction / (7 * 24)
    )

    df["time_period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df


def main():

    print("=" * 70)
    print("SPATIAL CONTEXT EXPERIMENT")
    print("=" * 70)

    print("\nLoading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(df):,}")

    df = create_features(df)

    base_features = [
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

    spatial_features = base_features + [
        "local_train_block_mean_rsrp"
    ]

    target = "rsrp_dbm"

    # ------------------------------------------------------------
    # Create spatial blocks
    # ------------------------------------------------------------

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

    blocks = np.asarray(
        df["spatial_block"].unique()
    )

    results = []

    for seed in SEEDS:

        print("\n" + "-" * 70)
        print(f"SEED {seed}")
        print("-" * 70)

        rng = np.random.default_rng(seed)

        shuffled_blocks = blocks.copy()
        rng.shuffle(shuffled_blocks)

        n_test_blocks = int(
            len(shuffled_blocks) * TEST_FRACTION
        )

        test_blocks = set(
            shuffled_blocks[:n_test_blocks]
        )

        test_mask = df["spatial_block"].isin(
            test_blocks
        )

        train_df = df.loc[~test_mask].copy()
        test_df = df.loc[test_mask].copy()

        print(
            f"Train rows: {len(train_df):,}"
        )
        print(
            f"Test rows:  {len(test_df):,}"
        )

        # --------------------------------------------------------
        # Calculate training block means
        # --------------------------------------------------------

        block_means = (
            train_df
            .groupby("spatial_block")[target]
            .mean()
        )

        overall_train_mean = train_df[target].mean()

        # --------------------------------------------------------
        # Build lookup of training block coordinates + means
        # --------------------------------------------------------

        train_blocks = (
            train_df[
                [
                    "lat_block",
                    "lon_block",
                    "spatial_block"
                ]
            ]
            .drop_duplicates("spatial_block")
            .copy()
        )

        train_blocks["block_mean"] = (
            train_blocks["spatial_block"]
            .map(block_means)
        )

        train_lookup = {
            (row.lat_block, row.lon_block):
                row.block_mean
            for row in train_blocks.itertuples()
        }

        # --------------------------------------------------------
        # For every held-out block, estimate its local signal
        # from nearby TRAINING blocks only.
        # --------------------------------------------------------

        test_blocks_df = (
            test_df[
                [
                    "lat_block",
                    "lon_block",
                    "spatial_block"
                ]
            ]
            .drop_duplicates("spatial_block")
        )

        local_values = {}

        for row in test_blocks_df.itertuples():

            neighbor_values = []

            for dlat in range(-2, 3):
                for dlon in range(-2, 3):

                    key = (
                        row.lat_block + dlat,
                        row.lon_block + dlon
                    )

                    if key in train_lookup:
                        neighbor_values.append(
                            train_lookup[key]
                        )

            if neighbor_values:
                local_values[row.spatial_block] = (
                    float(np.mean(neighbor_values))
                )
            else:
                local_values[row.spatial_block] = (
                    overall_train_mean
                )

        train_df["local_train_block_mean_rsrp"] = (
            train_df["spatial_block"]
            .map(block_means)
            .fillna(overall_train_mean)
        )

        test_df["local_train_block_mean_rsrp"] = (
            test_df["spatial_block"]
            .map(local_values)
            .fillna(overall_train_mean)
        )

        print(
            "Local spatial feature created from training blocks only."
        )

        # --------------------------------------------------------
        # Base model
        # --------------------------------------------------------

        base_model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )

        base_model.fit(
            train_df[base_features],
            train_df[target]
        )

        base_pred = base_model.predict(
            test_df[base_features]
        )

        base_rmse = np.sqrt(
            mean_squared_error(
                test_df[target],
                base_pred
            )
        )

        base_mae = mean_absolute_error(
            test_df[target],
            base_pred
        )

        base_r2 = r2_score(
            test_df[target],
            base_pred
        )

        # --------------------------------------------------------
        # Spatial-context model
        # --------------------------------------------------------

        spatial_model = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )

        spatial_model.fit(
            train_df[spatial_features],
            train_df[target]
        )

        spatial_pred = spatial_model.predict(
            test_df[spatial_features]
        )

        spatial_rmse = np.sqrt(
            mean_squared_error(
                test_df[target],
                spatial_pred
            )
        )

        spatial_mae = mean_absolute_error(
            test_df[target],
            spatial_pred
        )

        spatial_r2 = r2_score(
            test_df[target],
            spatial_pred
        )

        results.append({
            "seed": seed,
            "base_rmse_db": base_rmse,
            "spatial_rmse_db": spatial_rmse,
            "rmse_change_db": (
                spatial_rmse - base_rmse
            ),
            "base_mae_db": base_mae,
            "spatial_mae_db": spatial_mae,
            "base_r2": base_r2,
            "spatial_r2": spatial_r2,
        })

        print(
            f"Base RMSE:    {base_rmse:.3f} dB"
        )
        print(
            f"Spatial RMSE: {spatial_rmse:.3f} dB"
        )
        print(
            f"RMSE change:  "
            f"{spatial_rmse - base_rmse:+.3f} dB"
        )
        print(
            f"Base R2:      {base_r2:.3f}"
        )
        print(
            f"Spatial R2:   {spatial_r2:.3f}"
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("MEAN ± STD")
    print("=" * 70)

    for column, label in [
        ("base_rmse_db", "Base RMSE"),
        ("spatial_rmse_db", "Spatial-context RMSE"),
        ("rmse_change_db", "RMSE change"),
        ("base_mae_db", "Base MAE"),
        ("spatial_mae_db", "Spatial-context MAE"),
        ("base_r2", "Base R2"),
        ("spatial_r2", "Spatial-context R2"),
    ]:
        print(
            f"{label}: "
            f"{results_df[column].mean():.3f} ± "
            f"{results_df[column].std():.3f}"
        )

    print("\nSaved:")
    print(f"  {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
