# Winter Storm and Summer Price Pattern Summary

## Winter storm events
- Detected events: 53
- Top winter storm events by max LMP:
  - winter_storm_035: 2022-12-23 10:00:00-05:00 to 2022-12-27 23:00:00-05:00, max_lmp=500.00, min_temp_c=-13.43, max_wind_ms=11.92
  - winter_storm_001: 2018-01-01 00:00:00-05:00 to 2018-01-08 11:00:00-05:00, max_lmp=409.27, min_temp_c=-14.53, max_wind_ms=9.68
  - winter_storm_050: 2025-01-19 15:00:00-05:00 to 2025-01-23 10:00:00-05:00, max_lmp=389.72, min_temp_c=-11.12, max_wind_ms=6.18
  - winter_storm_040: 2024-01-16 05:00:00-05:00 to 2024-01-17 23:00:00-05:00, max_lmp=335.74, min_temp_c=-8.98, max_wind_ms=7.10
  - winter_storm_052: 2025-02-19 01:00:00-05:00 to 2025-02-21 10:00:00-05:00, max_lmp=272.13, min_temp_c=-5.48, max_wind_ms=8.75
  - winter_storm_036: 2022-12-28 04:00:00-05:00 to 2022-12-28 09:00:00-05:00, max_lmp=225.39, min_temp_c=-3.78, max_wind_ms=1.63
  - winter_storm_038: 2023-02-03 10:00:00-05:00 to 2023-02-04 09:00:00-05:00, max_lmp=222.86, min_temp_c=-8.07, max_wind_ms=8.92
  - winter_storm_046: 2025-01-08 06:00:00-05:00 to 2025-01-08 21:00:00-05:00, max_lmp=220.65, min_temp_c=-4.82, max_wind_ms=6.87
  - winter_storm_042: 2024-12-02 16:00:00-05:00 to 2024-12-02 23:00:00-05:00, max_lmp=215.56, min_temp_c=-0.57, max_wind_ms=4.63
  - winter_storm_047: 2025-01-09 06:00:00-05:00 to 2025-01-09 20:00:00-05:00, max_lmp=214.36, min_temp_c=-4.82, max_wind_ms=7.87

## Summer yearly pattern
- 2018: mean_lmp=32.99, p95_lmp=57.94, max_lmp=157.74, max_temp_c=34.15, high_price_hours=6
- 2019: mean_lmp=25.73, p95_lmp=43.19, max_lmp=71.55, max_temp_c=36.40, high_price_hours=0
- 2020: mean_lmp=23.13, p95_lmp=42.96, max_lmp=115.00, max_temp_c=36.75, high_price_hours=0
- 2021: mean_lmp=39.35, p95_lmp=74.64, max_lmp=161.66, max_temp_c=34.92, high_price_hours=12
- 2022: mean_lmp=102.93, p95_lmp=204.71, max_lmp=352.50, max_temp_c=35.40, high_price_hours=601
- 2023: mean_lmp=34.52, p95_lmp=63.85, max_lmp=309.16, max_temp_c=35.77, high_price_hours=13
- 2024: mean_lmp=39.30, p95_lmp=90.69, max_lmp=312.03, max_temp_c=37.12, high_price_hours=62
- 2025: mean_lmp=62.30, p95_lmp=154.92, max_lmp=678.09, max_temp_c=35.58, high_price_hours=184

## Winter yearly pattern
- winter 2018: mean_lmp=50.64, p95_lmp=161.12, max_lmp=409.27, min_temp_c=-14.53, high_price_hours=267
- winter 2019: mean_lmp=31.61, p95_lmp=50.63, max_lmp=189.32, min_temp_c=-11.40, high_price_hours=22
- winter 2020: mean_lmp=21.18, p95_lmp=31.28, max_lmp=69.37, min_temp_c=-6.03, high_price_hours=0
- winter 2021: mean_lmp=30.58, p95_lmp=53.12, max_lmp=182.69, min_temp_c=-5.65, high_price_hours=55
- winter 2022: mean_lmp=54.07, p95_lmp=104.15, max_lmp=199.50, min_temp_c=-8.60, high_price_hours=207
- winter 2023: mean_lmp=52.58, p95_lmp=137.31, max_lmp=500.00, min_temp_c=-13.43, high_price_hours=234
- winter 2024: mean_lmp=32.98, p95_lmp=79.46, max_lmp=335.74, min_temp_c=-8.98, high_price_hours=105
- winter 2025: mean_lmp=53.79, p95_lmp=132.65, max_lmp=389.72, min_temp_c=-11.12, high_price_hours=237

## Sudden weather events
- Summer sudden-heat events: 10
- Winter cold-wave events: 132
- Largest major-storm mean LMP uplifts vs pre-72h:
  - winter_storm_035: uplift=217.02
  - winter_storm_050: uplift=187.44
  - winter_storm_052: uplift=74.61
  - winter_storm_040: uplift=65.78
  - winter_storm_036: uplift=-134.10

## Forecast metrics by context
- test / other: best_mae_model=dmdc_residual_24h, mae=11.02, mse=521.99, rmse=22.85, r2=0.480, n=5113
- test / summer_all: best_mae_model=dmdc_residual_24h, mae=9.47, mse=553.24, rmse=23.52, r2=-0.302, n=3928
- test / summer_heat_price_event: best_mae_model=dmdc_residual_24h, mae=67.68, mse=10846.47, rmse=104.15, r2=0.047, n=354
- test / summer_sudden_heat: best_mae_model=seasonal_naive_168h, mae=7.44, mse=78.86, rmse=8.88, r2=0.540, n=15
- test / winter_all: best_mae_model=dmdc_residual_24h, mae=10.66, mse=413.14, rmse=20.33, r2=0.006, n=4409
- test / winter_cold_wave: best_mae_model=dmdc_residual_24h, mae=19.89, mse=1472.14, rmse=38.37, r2=0.645, n=369
- test / winter_storm: best_mae_model=dmdc_residual_24h, mae=29.60, mse=1871.56, rmse=43.26, r2=0.645, n=308
- validation / other: best_mae_model=dmdc_residual_24h, mae=8.54, mse=413.09, rmse=20.32, r2=0.122, n=3648
- validation / summer_all: best_mae_model=seasonal_naive_24h, mae=5.81, mse=176.67, rmse=13.29, r2=0.232, n=2122
- validation / summer_heat_price_event: best_mae_model=dmdc_residual_24h, mae=86.41, mse=12278.48, rmse=110.81, r2=-1.925, n=15
- validation / winter_all: best_mae_model=seasonal_naive_24h, mae=6.15, mse=157.86, rmse=12.56, r2=-0.300, n=2660
- validation / winter_cold_wave: best_mae_model=dmdc_residual_24h, mae=10.74, mse=582.58, rmse=24.14, r2=0.372, n=223
- validation / winter_storm: best_mae_model=dmdc_residual_24h, mae=45.24, mse=4587.85, rmse=67.73, r2=-1.006, n=20