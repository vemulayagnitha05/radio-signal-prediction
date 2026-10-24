import pandas as pd
import numpy as np


DATA_PATH = "data/scanner_data_lte.parquet"
BLOCK_SIZE = 0.002
RANDOM_STATE = 42
TEST_FRACTION = 0.20


def main():

    print("=" * 70)
    print("SPATIAL SPLIT VALIDATION")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Create spatial blocks
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reproduce the exact train/test split
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Basic split information
    # --------------------------------------------------------

    print("\nSpatial split")
    print("-" * 50)

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

    # --------------------------------------------------------
    # Cell overlap
    # --------------------------------------------------------

    train_cells = set(
        train_df["cell_id"].unique()
    )

    test_cells = set(
        test_df["cell_id"].unique()
    )

    seen_test_cells = (
        train_cells & test_cells
    )

    unseen_test_cells = (
        test_cells - train_cells
    )

    print("\nCell coverage")
    print("-" * 50)

    print(
        f"Unique train cells: "
        f"{len(train_cells):,}"
    )

    print(
        f"Unique test cells:  "
        f"{len(test_cells):,}"
    )

    print(
        f"Test cells seen in train: "
        f"{len(seen_test_cells):,}"
    )

    print(
        f"Unseen test cells: "
        f"{len(unseen_test_cells):,}"
    )

    if len(test_cells) > 0:
        print(
            f"Test cells seen percentage: "
            f"{100 * len(seen_test_cells) / len(test_cells):.2f}%"
        )

        print(
            f"Test cells unseen percentage: "
            f"{100 * len(unseen_test_cells) / len(test_cells):.2f}%"
        )

    # --------------------------------------------------------
    # Operator coverage
    # --------------------------------------------------------

    print("\nOperator coverage")
    print("-" * 50)

    train_ops = set(
        train_df["operator"].unique()
    )

    test_ops = set(
        test_df["operator"].unique()
    )

    print(
        f"Train operators: "
        f"{sorted(train_ops)}"
    )

    print(
        f"Test operators:  "
        f"{sorted(test_ops)}"
    )

    # --------------------------------------------------------
    # Channel coverage
    # --------------------------------------------------------

    print("\nChannel coverage")
    print("-" * 50)

    train_channels = set(
        train_df["channel_number"].unique()
    )

    test_channels = set(
        test_df["channel_number"].unique()
    )

    unseen_test_channels = (
        test_channels - train_channels
    )

    print(
        f"Train channels: "
        f"{len(train_channels)}"
    )

    print(
        f"Test channels:  "
        f"{len(test_channels)}"
    )

    print(
        f"Unseen test channels: "
        f"{len(unseen_test_channels)}"
    )

    if unseen_test_channels:
        print(
            f"Unseen channel IDs: "
            f"{sorted(unseen_test_channels)}"
        )

    # --------------------------------------------------------
    # Geographic separation
    # --------------------------------------------------------

    print("\nGeographic ranges")
    print("-" * 50)

    print(
        f"Train latitude: "
        f"{train_df['latitude'].min():.6f} "
        f"to "
        f"{train_df['latitude'].max():.6f}"
    )

    print(
        f"Test latitude:  "
        f"{test_df['latitude'].min():.6f} "
        f"to "
        f"{test_df['latitude'].max():.6f}"
    )

    print(
        f"Train longitude: "
        f"{train_df['longitude'].min():.6f} "
        f"to "
        f"{train_df['longitude'].max():.6f}"
    )

    print(
        f"Test longitude:  "
        f"{test_df['longitude'].min():.6f} "
        f"to "
        f"{test_df['longitude'].max():.6f}"
    )

    # --------------------------------------------------------
    # Save validation summary
    # --------------------------------------------------------

    summary = pd.DataFrame([{
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_blocks": train_df["spatial_block"].nunique(),
        "test_blocks": test_df["spatial_block"].nunique(),
        "train_cells": len(train_cells),
        "test_cells": len(test_cells),
        "seen_test_cells": len(seen_test_cells),
        "unseen_test_cells": len(unseen_test_cells),
        "seen_test_cell_pct": (
            100 * len(seen_test_cells) / len(test_cells)
            if len(test_cells) else 0
        ),
        "unseen_test_cell_pct": (
            100 * len(unseen_test_cells) / len(test_cells)
            if len(test_cells) else 0
        ),
        "train_operators": len(train_ops),
        "test_operators": len(test_ops),
        "unseen_test_channels": len(
            unseen_test_channels
        ),
    }])

    summary.to_csv(
        "results/split_validation.csv",
        index=False,
    )

    print(
        "\nValidation saved to:"
    )

    print(
        "results/split_validation.csv"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
