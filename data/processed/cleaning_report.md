# PJM Weather 24h Dataset Report

## Inputs
- PJM day-ahead DOM LMP: `PJM/pjm_dom_da_lmp_hrl_fixed.csv`
- PJM DOM load: `PJM/pjm_dom_load_hourly_2018_2025UTC.csv`
- Weather model-ready observations: `tianqi/Worked/weather_model_ready.csv`

## Output row counts
- Clean hourly table rows: 67,080
- 24h supervised table rows: 66,520

## Split counts
| split      |   rows | start                     | end                       |
|:-----------|-------:|:--------------------------|:--------------------------|
| test       |  14477 | 2024-01-01 00:00:00+00:00 | 2025-08-26 04:00:00+00:00 |
| train      |  43425 | 2018-01-08 05:00:00+00:00 | 2022-12-30 05:00:00+00:00 |
| validation |   8618 | 2023-01-01 04:00:00+00:00 | 2023-12-31 23:00:00+00:00 |

## Cleaning choices
- LMP gaps are time-interpolated where possible and marked by `lmp_missing`.
- Target columns use observed LMP only; rows with missing 24h targets are excluded from supervised splits.
- Load and continuous weather fields are interpolated over short gaps, then forward/back filled.
- Precipitation missing values are filled with zero and marked with missing indicators.
- Heat index and wind chill are filled with temperature when physically unavailable and marked by indicators.
- Price spikes are preserved; the target is not winsorized or clipped.