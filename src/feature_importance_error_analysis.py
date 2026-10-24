import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/scanner_data_lte.parquet"
OUTPUT_IMPORTANCE = "results/feature_importance.csv"
OUTPUT_ERROR = "results/error_analysis.csv"

BLOCK_SIZE = 0.002
TEST_FRACTION = 0.20
SEED = 42


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
    print("FEATURE IMPORTANCE + ERROR ANALYSIS")
    print("=" * 70)

    print("\nLoading dataset...")
    df = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(df):,}")

    df = create_features(df)

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

    target = "rsrp_dbm"

    # ------------------------------------------------------------
    # Spatial block split
    # ------------------------------------------------------------

    df["lat_block"] = np.floor(df["latitude"] / BLOCK_SIZE).astype(int)
    df["lon_block"] = np.floor(df["longitude"] / BLOCK_SIZE).astype(int)

    df["spatial_block"] = (
        df["lat_block"].astype(str)
        + "_"
        + df["lon_block"].astype(str)
    )

    blocks = df["spatial_block"].unique()

    rng = np.random.default_rng(SEED)
    rng.shuffle(blocks)

    n_test_blocks = int(len(blocks) * TEST_FRACTION)

    test_blocks = set(blocks[:n_test_blocks])

    test_mask = df["spatial_block"].isin(test_blocks)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    print("\nSpatial split:")
    print(f"  Train rows: {len(train_df):,}")
    print(f"  Test rows:  {len(test_df):,}")
    print(f"  Train blocks: {train_df['spatial_block'].nunique():,}")
    print(f"  Test blocks:  {test_df['spatial_block'].nunique():,}")

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    # ------------------------------------------------------------
    # Train canonical model
    # ------------------------------------------------------------

    print("\nTraining HistGradientBoostingRegressor...")

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=SEED,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nTest performance:")
    print(f"  RMSE: {rmse:.3f} dB")
    print(f"  MAE:  {mae:.3f} dB")
    print(f"  R2:   {r2:.3f}")

    # ------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------

    print("\nCalculating permutation importance...")

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": features,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })

    importance_df["importance_rank"] = (
        importance_df["importance_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT_IMPORTANCE,
        index=False
    )

    print("\nFeature importance:")
    print(
        importance_df[
            [
                "importance_rank",
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------

    test_df["prediction"] = predictions
    test_df["error_db"] = predictions - y_test.to_numpy()
    test_df["absolute_error_db"] = np.abs(test_df["error_db"])

    # RSRP strength bins
    bins = [-np.inf, -110, -100, -90, -80, -70, np.inf]
    labels = [
        "< -110",
        "-110 to -100",
        "-100 to -90",
        "-90 to -80",
        "-80 to -70",
        "> -70",
    ]

    test_df["rsrp_bin"] = pd.cut(
        test_df[target],
        bins=bins,
        labels=labels
    )

    error_summary = (
        test_df
        .groupby("rsrp_bin", observed=False)
        .agg(
            observations=("rsrp_dbm", "size"),
            mean_actual_rsrp_db=("rsrp_dbm", "mean"),
            mean_prediction_db=("prediction", "mean"),
            mean_error_db=("error_db", "mean"),
            mae_db=("absolute_error_db", "mean"),
            rmse_db=("error_db", lambda x: np.sqrt(np.mean(x ** 2))),
        )
        .reset_index()
    )

    error_summary.to_csv(
        OUTPUT_ERROR,
        index=False
    )

    print("\nError by RSRP strength:")
    print(error_summary.to_string(index=False))

    # ------------------------------------------------------------
    # Overall error distribution
    # ------------------------------------------------------------

    print("\nAbsolute error distribution:")
    print(
        test_df["absolute_error_db"].describe(
            percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
        ).to_string()
    )

    print("\nLargest absolute errors:")
    worst = (
        test_df[
            [
                "latitude",
                "longitude",
                "frequency_khz",
                "channel_number",
                "rsrp_dbm",
                "prediction",
                "error_db",
                "absolute_error_db",
            ]
        ]
        .sort_values("absolute_error_db", ascending=False)
        .head(10)
    )

    print(worst.to_string(index=False))

    print("\nSaved:")
    print(f"  {OUTPUT_IMPORTANCE}")
    print(f"  {OUTPUT_ERROR}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
