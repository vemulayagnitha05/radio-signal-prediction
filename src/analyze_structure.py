import pandas as pd

DATA_PATH = "data/scanner_data_lte.parquet"

df = pd.read_parquet(DATA_PATH)

print("=" * 70)
print("LTE DATASET STRUCTURE ANALYSIS")
print("=" * 70)

print("\n[1] Measurement IDs")

n_measurements = df["measurement_id"].nunique()
rows_per_measurement = df.groupby("measurement_id").size()

print(f"Unique measurement IDs: {n_measurements:,}")
print("Rows per measurement:")
print(rows_per_measurement.describe())

print("\n[2] Operator distribution")

print(df["operator"].value_counts().sort_index())

print("\n[3] Channel distribution")

print(
    df.groupby("channel_number")
    .agg(
        rows=("channel_number", "size"),
        frequency_khz=("frequency_khz", "first"),
        operators=("operator", "nunique"),
        cells=("cell_id", "nunique"),
    )
    .sort_index()
)

print("\n[4] Cell distribution")

cell_counts = df.groupby("cell_id").size()

print(f"Unique cells: {cell_counts.size:,}")
print(cell_counts.describe())

print("\nTop 20 cells by number of observations:")
print(cell_counts.sort_values(ascending=False).head(20))

print("\n[5] Operator-cell structure")

operator_cells = df.groupby("operator")["cell_id"].nunique()
print(operator_cells)

print("\n[6] Geographic measurements")

unique_coordinates = df[
    ["latitude", "longitude"]
].drop_duplicates()

print(f"Unique coordinate pairs: {len(unique_coordinates):,}")

print("\n[7] Temporal structure")

time_sorted = df["time"].sort_values()
time_diffs = time_sorted.diff().dropna().dt.total_seconds()

print("Time difference statistics between consecutive rows:")
print(time_diffs.describe())

print("\n[8] RSRP by operator")

print(
    df.groupby("operator")["rsrp_dbm"]
    .agg(["count", "mean", "std", "min", "median", "max"])
)

print("\n[9] RSRP by channel")

print(
    df.groupby("channel_number")["rsrp_dbm"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .sort_index()
)

print("\n" + "=" * 70)
print("STRUCTURE ANALYSIS COMPLETE")
print("=" * 70)
