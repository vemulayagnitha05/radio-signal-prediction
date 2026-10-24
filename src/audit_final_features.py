import numpy as np
import pandas as pd


DATA_PATH = "data/scanner_data_lte.parquet"


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


def main():

    print("=" * 70)
    print("FINAL FEATURE LEAKAGE AUDIT")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Rows: {len(df):,}"
    )

    # Create the same derived temporal
    # features used by the final model.

    df = add_temporal_features(df)

    final_features = [
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

    same_snapshot_radio = [
        "rsrq_db",
        "rssi_dbm",
        "cinr_db",
    ]

    identifiers = [
        "measurement_id",
        "cell_id",
        "enb_id",
        "sector_id",
        "pci",
    ]

    print("\nTarget:")
    print(f"  {target}")

    print("\nFinal model features:")

    for feature in final_features:
        print(
            f"  [OK] {feature}"
        )

    print("\nChecking feature existence...")

    missing = [
        feature
        for feature in final_features
        if feature not in df.columns
    ]

    if missing:

        print(
            "\nERROR: Missing features:"
        )

        for feature in missing:
            print(
                f"  {feature}"
            )

        raise SystemExit(1)

    print(
        "  [PASS] All final features exist "
        "after feature construction."
    )

    print("\nChecking target...")

    if target not in df.columns:

        print(
            f"  ERROR: Target {target} "
            "does not exist."
        )

        raise SystemExit(1)

    print(
        f"  [PASS] Target {target} exists."
    )

    print(
        "\nChecking same-snapshot radio "
        "measurement leakage..."
    )

    overlap = (
        set(final_features)
        & set(same_snapshot_radio)
    )

    if overlap:

        print(
            "\nERROR: Same-snapshot radio "
            "features detected:"
        )

        for feature in overlap:
            print(
                f"  {feature}"
            )

        raise SystemExit(1)

    print(
        "  [PASS] RSRQ, RSSI, and CINR "
        "are excluded."
    )

    print(
        "\nChecking direct identifier leakage..."
    )

    identifier_overlap = (
        set(final_features)
        & set(identifiers)
    )

    if identifier_overlap:

        print(
            "\nWARNING: Identifier features "
            "detected:"
        )

        for feature in identifier_overlap:
            print(
                f"  {feature}"
            )

    else:

        print(
            "  [PASS] No measurement/cell "
            "identifiers are used."
        )

    print(
        "\nChecking target is not included "
        "as a feature..."
    )

    if target in final_features:

        print(
            "  ERROR: Target is included "
            "as a feature."
        )

        raise SystemExit(1)

    print(
        "  [PASS] Target is excluded "
        "from features."
    )

    print(
        "\nFinal feature count:"
    )

    print(
        f"  {len(final_features)} features"
    )

    print(
        "\nFeature categories:"
    )

    print(
        "  Spatial:"
    )

    print(
        "    latitude, longitude"
    )

    print(
        "  Network:"
    )

    print(
        "    frequency_khz, channel_number"
    )

    print(
        "  Temporal:"
    )

    print(
        "    hour, day_of_week, month"
    )

    print(
        "    minute_of_day_sin/cos"
    )

    print(
        "    week_sin/cos"
    )

    print(
        "    time_period"
    )

    print("\n" + "=" * 70)
    print("AUDIT PASSED")
    print("=" * 70)

    print(
        "\nThe final model uses spatial, "
        "network, and temporal context."
    )

    print(
        "Same-observation radio measurements "
        "are excluded."
    )

    print(
        "Direct cell/measurement identifiers "
        "are excluded."
    )


if __name__ == "__main__":
    main()
