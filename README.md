# LTE Radio Signal Strength Prediction

A real-data machine learning project for predicting LTE Reference Signal Received Power (RSRP) from spatial, network, and temporal context using the Vienna LTE/5G drive-test dataset.

## Objective

Predict LTE RSRP (`rsrp_dbm`) at an unseen geographic location using information available before observing the target radio measurement.

The main question is:

> How accurately can LTE signal strength be predicted at geographic locations that were not used for training?

## Dataset

The project uses LTE scanner data from the Vienna LTE/5G drive-test dataset.

Working dataset:

- 529,724 LTE observations
- 193,457 unique measurement IDs
- 137,296 unique coordinate pairs
- 5,679 unique LTE cells
- 3 operators
- 13 LTE channels
- Measurements spanning June 2024 to March 2025

Target:

`rsrp_dbm` — LTE Reference Signal Received Power in dBm.

## Final Features

The final model uses 12 leakage-safe features.

### Spatial

- latitude
- longitude

### Network

- frequency_khz
- channel_number

### Temporal

- hour
- day_of_week
- month
- minute_of_day_sin
- minute_of_day_cos
- week_sin
- week_cos
- time_period

## Leakage Control

Same-observation radio measurements were deliberately excluded:

- RSRQ
- RSSI
- CINR

These measurements are collected alongside RSRP and would make the prediction task artificially easy.

Measurement IDs and cell IDs were also excluded from the final feature set.

The final feature audit confirmed that the target and same-snapshot radio measurements were not used as predictors.

## Evaluation Method

Random row-level train/test splitting was avoided because nearby observations can be strongly correlated geographically.

Instead, latitude and longitude were divided into approximately 0.002-degree spatial blocks.

For each of five random seeds:

1. Spatial blocks were shuffled.
2. 20% of spatial blocks were held out.
3. Observations inside those blocks formed the test set.
4. The model was trained only on the remaining blocks.

Seeds:

`1, 21, 42, 84, 123`

Across the five splits, the median distance from sampled test observations to the nearest training-block center was approximately 0.124 km.

This evaluates local geographic generalization rather than random row memorization.

## Baselines

All baselines use the same spatial holdout methodology.

| Model | RMSE (dB) | R2 |
|---|---:|---:|
| Global mean | 15.862 +/- 0.423 | -0.001 +/- 0.001 |
| Operator + channel mean | 14.404 +/- 0.370 | 0.175 +/- 0.009 |
| Final ML model | 11.948 +/- 0.087 | 0.431 +/- 0.024 |

The final ML model reduces RMSE by:

- 3.914 dB versus the global mean
- 2.456 dB versus the operator + channel baseline

## Final Model

The final model uses histogram-based gradient boosting regression with squared-error loss.

Performance across five spatial holdout seeds:

- RMSE: 11.948 +/- 0.087 dB
- R2: 0.431 +/- 0.024

## Model Diagnostics

Permutation importance showed that frequency and spatial coordinates were the strongest contributors to predictive performance.

The model also showed regression toward the mean:

- weak signals tended to be predicted too strongly
- strong signals tended to be predicted too weakly

This indicates that the model captures broad spatial and frequency-dependent structure but has difficulty reproducing extreme signal conditions.

## Additional Experiments

### Historical Cell-Mean Encoding

A training-only historical mean RSRP was calculated for each cell.

This increased RMSE from approximately 12.03 dB to 12.27 dB across the five spatial splits.

The feature was rejected.

### Local Spatial Context

A training-only local spatial-block mean was tested as an additional feature.

It consistently worsened performance across all five seeds and was rejected.

### Rich Temporal Features

Cyclical time features produced a modest improvement over the simpler temporal representation.

### Categorical Encoding

A dense one-hot encoding approach involving network identifiers became computationally expensive and was not used in the final model.

## Reproducibility

Core analysis scripts are located in `src/`.

Key scripts include:

- `inspect_data.py`
- `analyze_structure.py`
- `check_cell_behavior.py`
- `validate_split.py`
- `evaluate_baselines.py`
- `audit_final_features.py`
- `analyze_spatial_separation.py`
- `feature_importance_error_analysis.py`
- `create_final_summary.py`

Results are stored in `results/`.

Final summary:

`results/final_model_summary.csv`

## Project Structure

radio-signal-prediction/
├── data/
│   └── scanner_data_lte.parquet
├── models/
├── notebooks/
├── results/
└── src/

## Key Takeaway

Spatial and frequency context provide substantial predictive value for LTE RSRP.

Using spatial holdout evaluation, the final gradient-boosting model achieves approximately 12 dB RMSE on unseen geographic blocks, with stable performance across five independent spatial splits.

## Experimental Notes

The repository also contains exploratory experiments that were not selected for the final model.

- `evaluate_categorical.py` tests dense one-hot encoding of categorical network identifiers. The approach became computationally expensive and was not used in the final model.
- `evaluate_target_models.py` compares squared-error and absolute-error training losses. The experiment was not completed across all five spatial splits, so its results are not used for the headline performance metrics.

The reported final performance comes from the reproducible five-seed evaluation in `evaluate_baselines.py`.
