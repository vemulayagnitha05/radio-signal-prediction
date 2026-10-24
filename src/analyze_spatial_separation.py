import numpy as np
import pandas as pd


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


def calculate_separation(train_df, test_df):

    # Convert latitude/longitude differences
    # to approximate kilometers.
    #
    # 1 degree latitude ~= 111 km
    # longitude distance adjusted by latitude.

    train_lat = train_df[
        "latitude"
    ].to_numpy()

    train_lon = train_df[
        "longitude"
    ].to_numpy()

    test_lat = test_df[
        "latitude"
    ].to_numpy()

    test_lon = test_df[
        "longitude"
    ].to_numpy()

    # To avoid comparing every test row against
    # every training row, compare test points to
    # training spatial-block centers.

    train_blocks = (
        train_df[
            [
                "spatial_block",
                "latitude",
                "longitude",
            ]
        ]
        .groupby("spatial_block")
        .mean()
    )

    train_block_lat = train_blocks[
        "latitude"
    ].to_numpy()

    train_block_lon = train_blocks[
        "longitude"
    ].to_numpy()

    # Sample test points if necessary.
    # This keeps the calculation lightweight.

    rng = np.random.default_rng(12345)

    max_points = 5000

    if len(test_lat) > max_points:

        sample_idx = rng.choice(
            len(test_lat),
            size=max_points,
            replace=False,
        )

        test_lat_sample = test_lat[
            sample_idx
        ]

        test_lon_sample = test_lon[
            sample_idx
        ]

    else:

        test_lat_sample = test_lat
        test_lon_sample = test_lon

    distances = []

    for lat, lon in zip(
        test_lat_sample,
        test_lon_sample,
    ):

        lat_diff = (
            train_block_lat - lat
        ) * 111.0

        lon_diff = (
            train_block_lon - lon
        ) * 111.0 * np.cos(
            np.radians(lat)
        )

        distance_km = np.sqrt(
            lat_diff ** 2
            + lon_diff ** 2
        )

        distances.append(
            distance_km.min()
        )

    distances = np.asarray(
        distances
    )

    return {
        "min_km": distances.min(),
        "p25_km": np.percentile(
            distances,
            25,
        ),
        "median_km": np.median(
            distances
        ),
        "p75_km": np.percentile(
            distances,
            75,
        ),
        "max_km": distances.max(),
        "mean_km": distances.mean(),
    }


def main():

    print("=" * 70)
    print("SPATIAL SEPARATION DIAGNOSTIC")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

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

        train_blocks = (
            train_df[
                "spatial_block"
            ].nunique()
        )

        test_blocks = (
            test_df[
                "spatial_block"
            ].nunique()
        )

        print(
            f"Train blocks: {train_blocks}"
        )

        print(
            f"Test blocks:  {test_blocks}"
        )

        separation = calculate_separation(
            train_df,
            test_df,
        )

        print(
            "\nDistance from each sampled "
            "test point to nearest training block:"
        )

        print(
            f"Minimum: {separation['min_km']:.3f} km"
        )

        print(
            f"25th percentile: "
            f"{separation['p25_km']:.3f} km"
        )

        print(
            f"Median: "
            f"{separation['median_km']:.3f} km"
        )

        print(
            f"75th percentile: "
            f"{separation['p75_km']:.3f} km"
        )

        print(
            f"Mean: "
            f"{separation['mean_km']:.3f} km"
        )

        print(
            f"Maximum: "
            f"{separation['max_km']:.3f} km"
        )

        results.append({
            "seed": seed,
            "train_blocks": train_blocks,
            "test_blocks": test_blocks,
            **separation,
        })

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("SPATIAL SEPARATION SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nMean ± Std")

    print(
        f"Median separation: "
        f"{results_df['median_km'].mean():.3f}"
        f" ± "
        f"{results_df['median_km'].std():.3f} km"
    )

    print(
        f"Mean separation: "
        f"{results_df['mean_km'].mean():.3f}"
        f" ± "
        f"{results_df['mean_km'].std():.3f} km"
    )

    output_path = (
        "results/spatial_separation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
