import pandas as pd

DATA_PATH = "data/scanner_data_lte.parquet"

df = pd.read_parquet(DATA_PATH)

print("=" * 70)
print("CELL BEHAVIOR ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Cell -> operator consistency
# ------------------------------------------------------------
cell_operator_counts = (
    df.groupby("cell_id")["operator"]
    .nunique()
)

print("\n[1] Cell/operator consistency")
print(
    cell_operator_counts.value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# 2. Cell -> channel consistency
# ------------------------------------------------------------
cell_channel_counts = (
    df.groupby("cell_id")["channel_number"]
    .nunique()
)

print("\n[2] Cell/channel consistency")
print(
    cell_channel_counts.describe()
)

print("\nCells using more than one channel:")
print(
    (cell_channel_counts > 1).sum()
)

# ------------------------------------------------------------
# 3. Cell geographic extent
# ------------------------------------------------------------
cell_geo = (
    df.groupby("cell_id")
    .agg(
        observations=("cell_id", "size"),
        min_lat=("latitude", "min"),
        max_lat=("latitude", "max"),
        min_lon=("longitude", "min"),
        max_lon=("longitude", "max"),
    )
)

cell_geo["lat_range"] = (
    cell_geo["max_lat"] - cell_geo["min_lat"]
)

cell_geo["lon_range"] = (
    cell_geo["max_lon"] - cell_geo["min_lon"]
)

print("\n[3] Cell geographic extent")
print(
    cell_geo[
        ["observations", "lat_range", "lon_range"]
    ].describe()
)

print("\nTop 20 cells by latitude span:")
print(
    cell_geo.sort_values("lat_range", ascending=False)
    .head(20)
)

print("\nTop 20 cells by longitude span:")
print(
    cell_geo.sort_values("lon_range", ascending=False)
    .head(20)
)

# ------------------------------------------------------------
# 4. Duplicate coordinate / cell combinations
# ------------------------------------------------------------
print("\n[4] Coordinate-cell combinations")

coordinate_cell = df[
    ["latitude", "longitude", "cell_id"]
].drop_duplicates()

print(
    f"Unique coordinate/cell combinations: "
    f"{len(coordinate_cell):,}"
)

# ------------------------------------------------------------
# 5. Same coordinate, multiple cells
# ------------------------------------------------------------
coordinate_counts = (
    df.groupby(["latitude", "longitude"])["cell_id"]
    .nunique()
)

print("\n[5] Cells visible at each coordinate")
print(coordinate_counts.describe())

print("\n" + "=" * 70)
print("CELL ANALYSIS COMPLETE")
print("=" * 70)
