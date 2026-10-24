import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/scanner_data_lte.parquet"

RANDOM_STATE = 42
TEST_FRACTION = 0.20
BLOCK_SIZE = 0.002


def add_features(df):
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

    operator_map = {
        operator: i
        for i, operator
        in enumerate(sorted(df["operator"].unique()))
    }

    df["operator_code"] = (
        df["operator"]
        .map(operator_map)
        .astype("int16")
    )

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

    return df


def spatial_split(df):
    blocks = np.array(
        df["spatial_block"].unique(),
        dtype=str,
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

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


def evaluate(
    name,
    model,
    X_test,
    y_test,
):

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

    print(f"\n{name}")
    print("-" * 50)
    print(f"RMSE: {rmse:.3f} dB")
    print(f"MAE:  {mae:.3f} dB")
    print(f"R²:   {r2:.3f}")

    return {
        "model": name,
        "rmse_db": rmse,
        "mae_db": mae,
        "r2": r2,
    }


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    name,
):

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
    )

    result = evaluate(
        name,
        model,
        X_test,
        y_test,
    )

    return model, result


def main():

    print("=" * 70)
    print("LTE RSRP SPATIAL PREDICTION")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    df = add_features(df)

    train_df, test_df = spatial_split(
        df
    )

    print("\nSpatial split")
    print("-" * 50)

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Test rows:  {len(test_df):,}"
    )

    print(
        f"Train fraction: "
        f"{len(train_df) / len(df):.3f}"
    )

    print(
        f"Test fraction:  "
        f"{len(test_df) / len(df):.3f}"
    )

    print(
        f"Train blocks: "
        f"{train_df['spatial_block'].nunique():,}"
    )

    print(
        f"Test blocks: "
        f"{test_df['spatial_block'].nunique():,}"
    )

    y_train = train_df[
        "rsrp_dbm"
    ]

    y_test = test_df[
        "rsrp_dbm"
    ]

    results = []

    baseline_prediction = np.full(
        len(y_test),
        y_train.mean(),
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_prediction,
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_prediction,
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_prediction,
    )

    print("\nBaseline: Training Mean")
    print("-" * 50)

    print(
        f"Training mean: "
        f"{y_train.mean():.3f} dBm"
    )

    print(
        f"RMSE: {baseline_rmse:.3f} dB"
    )

    print(
        f"MAE:  {baseline_mae:.3f} dB"
    )

    print(
        f"R²:   {baseline_r2:.3f}"
    )

    results.append({
        "model": "Baseline: Training Mean",
        "rmse_db": baseline_rmse,
        "mae_db": baseline_mae,
        "r2": baseline_r2,
    })

    features_1 = [
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

    X_train_1 = train_df[
        features_1
    ]

    X_test_1 = test_df[
        features_1
    ]

    _, result_1 = train_model(
        X_train_1,
        y_train,
        X_test_1,
        y_test,
        "Model 1: Spatial + Network",
    )

    results.append(
        result_1
    )

    features_2 = features_1 + [
        "cell_id",
        "enb_id",
        "sector_id",
        "pci",
    ]

    X_train_2 = train_df[
        features_2
    ]

    X_test_2 = test_df[
        features_2
    ]

    _, result_2 = train_model(
        X_train_2,
        y_train,
        X_test_2,
        y_test,
        "Model 2: Cell-aware",
    )

    results.append(
        result_2
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        "results/model_comparison.csv"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
