# Loss Function and Metric Notes

## Regression losses
- MAE is robust to occasional price spikes and is easier to interpret in $/MWh.
- MSE and RMSE penalize large spike misses more heavily, so they are useful for winter storm and summer scarcity periods.
- R2 measures explained variance relative to predicting the sample mean. Negative R2 means the model is worse than that mean baseline for that evaluated group.
- Bias is the mean signed error; negative bias means under-forecasting.

## High-price ROC/AUC setup
- The high-price class is defined as observed LMP >= 78.7659.
- Forecasted price is used as the classification score. This evaluates ranking ability for high-price hours, not calibrated probability.

## Best holdout MAE by split
- test: dmdc_residual_24h MAE=12.4959, MSE=802.0480, R2=0.5360
- validation: dmdc_residual_24h MAE=7.4987, MSE=306.7262, R2=0.2358

## Best expanding-window CV MAE by fold
- cv_2020: seasonal_naive_24h MAE=3.0800, MSE=28.3905, R2=0.5848
- cv_2021: dmdc_residual_24h MAE=6.8220, MSE=207.8948, R2=0.6111
- cv_2022: dmdc_residual_24h MAE=16.4879, MSE=949.3154, R2=0.6504