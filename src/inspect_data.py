import pandas as pd

DATA_PATH = "data/scanner_data_lte.parquet"

print("=" * 70)
print("VIENNA LTE DATASET INSPECTION")
print("=" * 70)

df = pd.read_parquet(DATA_PATH)

print("\n[1] Dataset shape")
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n[2] Columns")
for col in df.columns:
    print(f"  - {col}")

print("\n[3] Data types")
print(df.dtypes)

print("\n[4] Missing values")
missing = df.isna().sum()
print(missing)

print("\n[5] First 5 rows")
print(df.head().to_string())

print("\n[6] RSRP statistics")
print(df["rsrp_dbm"].describe())

print("\n[7] RSRQ statistics")
print(df["rsrq_db"].describe())

print("\n[8] CINR statistics")
print(df["cinr_db"].describe())

print("\n[9] Categorical / identifier cardinality")

for col in [
    "operator",
    "channel_number",
    "pci",
    "cell_id",
    "enb_id",
    "sector_id",
]:
    print(f"{col:20s}: {df[col].nunique():,} unique values")

print("\n[10] Time range")
print(f"Start: {df['time'].min()}")
print(f"End:   {df['time'].max()}")

print("\n[11] Geographic range")
print(f"Latitude:  {df['latitude'].min():.6f} -> {df['latitude'].max():.6f}")
print(f"Longitude: {df['longitude'].min():.6f} -> {df['longitude'].max():.6f}")

print("\n[12] RSRP quantiles")
print(
    df["rsrp_dbm"]
    .quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
)

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
