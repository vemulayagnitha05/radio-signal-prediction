import pandas as pd

global_rmse = 15.862
group_rmse = 14.404
ml_rmse = 11.948

summary = pd.DataFrame([
    {
        "experiment": "Global mean baseline",
        "rmse_mean_db": global_rmse,
        "rmse_std_db": 0.423,
        "r2_mean": -0.001,
        "r2_std": 0.001,
    },
    {
        "experiment": "Operator + channel baseline",
        "rmse_mean_db": group_rmse,
        "rmse_std_db": 0.370,
        "r2_mean": 0.175,
        "r2_std": 0.009,
    },
    {
        "experiment": "Final ML model",
        "rmse_mean_db": ml_rmse,
        "rmse_std_db": 0.087,
        "r2_mean": 0.431,
        "r2_std": 0.024,
        "improvement_vs_global_db": global_rmse - ml_rmse,
        "improvement_vs_group_db": group_rmse - ml_rmse,
    },
])

output = "results/final_model_summary.csv"
summary.to_csv(output, index=False)

print(summary.to_string(index=False))
print()
print(f"ML RMSE improvement vs global mean: {global_rmse - ml_rmse:.3f} dB")
print(f"ML RMSE improvement vs operator+channel: {group_rmse - ml_rmse:.3f} dB")
print(f"Saved: {output}")
