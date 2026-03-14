# Impact of U.S. Summer Heatwaves and Winter Blizzards on PJM Electricity Prices
The Project for DSCI 441

## Project Status
This is an ongoing research project. The README will be updated as the empirical model, robustness checks, and final results are refined.

## Project Summary
This project studies how extreme summer heatwaves and winter blizzards affect wholesale electricity prices in PJM, with a primary focus on the Dominion (DOM) zone and PJM market outcomes over the 2018–2025 period. The core question is how severe weather shocks change electricity demand conditions, tighten system operations, and translate into abnormal movements in day-ahead and real-time locational marginal prices (LMPs). More broadly, the project aims to quantify the relationship between extreme weather, load stress, and electricity price volatility in a major U.S. wholesale power market.

The project is motivated by the growing importance of weather-driven reliability stress in power systems. Summer heatwaves can sharply increase cooling demand and peak load, while winter blizzards can disrupt generation availability, fuel delivery, and transmission conditions. These events may produce large and asymmetric impacts on electricity prices. By identifying and estimating these effects, the project contributes to ongoing discussions about grid resilience, market design, and the economic consequences of extreme weather in electricity systems.

## Methodology
The empirical strategy combines time-series econometrics with event-based analysis. First, the project constructs an hourly dataset by merging PJM day-ahead LMPs, real-time LMPs, DOM zonal load, and weather variables. The weather block includes processed hourly station data designed to capture the thermal and meteorological conditions most relevant to electricity demand and system stress. The analysis window covers 2018–2025.

Second, the project estimates SARIMAX-style models to explain electricity prices using both seasonal dynamics and exogenous drivers. The baseline specification incorporates hourly and weekly seasonality, load conditions, and weather-related variables such as temperature-based measures and other meteorological indicators. This framework is used to distinguish ordinary seasonal price behavior from abnormal price responses associated with extreme weather conditions.

Third, the project uses event-study logic to examine major summer heatwave periods and major winter blizzard episodes. Event windows are compared against model-based baselines in order to estimate abnormal price movements, persistence, and differences between day-ahead and real-time markets. Model diagnostics include stationarity tests, residual autocorrelation checks, heteroskedasticity checks, lag selection, and robustness tests across alternative weather definitions and peak-hour specifications.

## Research Questions
1. How do U.S. summer heatwaves affect PJM day-ahead and real-time electricity prices?
2. How do winter blizzards affect PJM electricity prices, and do they produce different price dynamics from heatwaves?
3. Are weather-driven price effects stronger during peak hours or in real-time markets?
4. How much of PJM price variation can be explained by extreme weather after controlling for seasonality and load conditions?

## Data Sources
The project uses the following data sources and processed files:

### PJM market and load data
- **Hourly real-time LMPs**: `pjm_dom_rt_lmp_hrl_merged.csv`
- **Hourly day-ahead LMPs**: `pjm_dom_da_lmp_hrl_fixed.csv`
- **Hourly DOM load**: `pjm_dom_load_hourly_2018_2025UTC.csv`

These datasets are based on PJM market and load records and are used to measure hourly price behavior and zonal load conditions.

### Weather data
- **Processed model-ready weather data**: `weather_model_ready.csv`
- **Station-level hourly weather coverage**: `weather_hours_per_station.csv`
- **Station-level span/coverage summary**: `weather_span_per_station.csv`

These weather files are used to construct the exogenous meteorological variables for the econometric models.

## Expected Contribution
This project contributes to the literature on electricity market behavior under extreme weather by:
- estimating the effect of heatwaves and blizzards on PJM electricity prices,
- comparing day-ahead and real-time price responses,
- linking weather shocks to system load stress and price volatility, and
- providing an empirical basis for discussions of resilience, planning, and market response under increasingly frequent extreme weather events.

## Repository Contents
A typical repository structure for this project may include:
- `data/` for cleaned and merged datasets
- `notebooks/` or `scripts/` for preprocessing and econometric analysis
- `figures/` for event-study plots, time-series charts, and diagnostics
- `output/` for regression tables and model results

## References
1. PJM Interconnection. **Data Miner 2** and related PJM market/load data resources.
2. NOAA Integrated Surface Database (ISD), used for hourly weather observations.
3. Majumder, S., Xie, L., and Aravena, I. **An Econometric Analysis of Large Flexible Cryptocurrency-mining Consumers in Electricity Markets**.
4. PJM Interconnection. **Large Load Additions: PJM Conceptual Proposal and Request for Member Feedback** (2025).
5. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., and Ljung, G. M. *Time Series Analysis: Forecasting and Control*.
6. Wooldridge, J. M. *Introductory Econometrics: A Modern Approach*.

## Notes
This repository currently focuses only on the PJM portion of the broader research agenda, specifically the impact of U.S. summer heatwaves and winter blizzards on PJM electricity prices. Additional model details, event definitions, estimation results, and robustness checks will be added in later updates.
