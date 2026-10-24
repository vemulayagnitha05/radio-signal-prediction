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

    test_mask = df["spatial_block"].isin(test_blocks)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    return train_df, test_df


def add_features(df):

    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    operator_map = {
        "A": 0,
        "B": 1,
        "C": 2,
    }

    df["operator_code"] = (
        df["operator"].map(operator_map)
    )

    return df


def evaluate_model(train_df, test_df, features):

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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
    print("SPATIAL SEED STABILITY EXPERIMENT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(
        f"Rows loaded: {len(df):,}"
    )

    df = add_features(df)

    features = [
        "latitude",
        "longitude",
        "frequency_khz",
        "channel_number",
        "operator_code",
        "hour",
        "day_of_week",
        "month",
        "hour_sin",
        "hour_cos",
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

        print(
            f"Train blocks: "
            f"{train_df['spatial_block'].nunique():,}"
        )

        print(
            f"Test blocks: "
            f"{test_df['spatial_block'].nunique():,}"
        )

        rmse, mae, r2 = evaluate_model(
            train_df,
            test_df,
            features,
        )

        print(
            f"RMSE: {rmse:.3f} dB"
        )

        print(
            f"MAE:  {mae:.3f} dB"
        )

        print(
            f"R²:   {r2:.3f}"
        )

        results.append({
            "seed": seed,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_blocks": train_df[
                "spatial_block"
            ].nunique(),
            "test_blocks": test_df[
                "spatial_block"
            ].nunique(),
            "rmse_db": rmse,
            "mae_db": mae,
            "r2": r2,
        })

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("SEED STABILITY SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "seed",
                "rmse_db",
                "mae_db",
                "r2",
            ]
        ].to_string(index=False)
    )

    print("\nMean ± Std")
    print("-" * 50)

    print(
        f"RMSE: "
        f"{results_df['rmse_db'].mean():.3f} "
        f"± "
        f"{results_df['rmse_db'].std():.3f} dB"
    )

    print(
        f"MAE:  "
        f"{results_df['mae_db'].mean():.3f} "
        f"± "
        f"{results_df['mae_db'].std():.3f} dB"
    )

    print(
        f"R²:   "
        f"{results_df['r2'].mean():.3f} "
        f"± "
        f"{results_df['r2'].std():.3f}"
    )

    output_path = (
        "results/seed_stability.csv"
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
