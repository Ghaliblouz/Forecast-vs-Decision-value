Electricity Price Forecast Quality vs. Decision Value
=====================================================
Progress Package — September 2026
Paper draft: [Overleaf link to be shared separately]


NOTEBOOKS (run order)
---------------------

1. draft_data.ipynb
   - Benchmark data exploration (2015-2018, 5 markets)
   - Price distributions, seasonal structure, tail concentration
   - Summary stats: mean, std, skewness, kurtosis per market

2. train_models.ipynb
   - Retrains LEAR (4 calibration windows) on Zenodo benchmark data
   - Validates MAE against published values (all within ±5%)
   - Also trains LightGBM (underperformed) and DNN (via Kaggle)

3. cross_market_decision.ipynb
   - Core analysis: regret vs lambda for DNN and LEAR ensembles
   - Diebold-Mariano tests with HAC correction
   - Key result: MAE p=0.14 (not significant), regret at lambda>=0.2 p=0.002
   - Cross-market replication (all 5 markets)
   - Forward premium sensitivity (0%, 5%, 10%)

4. entso_data_analysis.ipynb
   - ENTSO-E 2019-2023 data exploration (FR, BE, NP, DE)
   - Five price regimes: 2019 normal, 2020 COVID, 2021 ramp, 2022 crisis, 2023 post-crisis
   - Cross-market correlation, kurtosis comparison to benchmark

5. lear_new_data.ipynb
   - Robustness extension: AR-LEAR (price-lags only) on ENTSO-E 2022-2023
   - Regret grows with lambda (consistent with benchmark)
   - Caveat: confounded by model quality (no exogenous features) and price regime


FIGURES
-------

price_distributions.png       Histograms of realised prices, all 5 markets
error_by_decile_all.png       Signed error by price decile — shows systematic under-forecasting in tail
regret_vs_lambda_NP.png       Regret vs risk aversion (Nord Pool) — core finding
cross_market_regret_dm.png    Regret + DM p-values across all 5 markets
cross_market_centrepiece.png  Summary figure: accuracy vs value divergence
sensitivity_premium_NP.png    Robustness to forward premium assumption
new_period_regret_comparison.png  ENTSO-E 2022-2023 vs benchmark comparison


MODEL OUTPUTS
-------------

benchmark_forecasts/    Published DNN + LEAR forecasts from Lago et al. (2021)
                        5 CSVs, one per market, ~17,472 hours each

retrained_forecasts/    Our retrained models:
                        - {market}_LEAR_w{56,84,1092,1456}_zenodo.csv  (LEAR per window)
                        - {market}_LEAR_zenodo_baseline.csv            (LEAR ensemble)
                        - {market}_ARLEAR_entso_2022_2023.csv          (AR-LEAR on new data)
                        - NP_LGBM_zenodo_baseline.csv                  (LightGBM, underperformed)


DATA SOURCES (not included, publicly available)
-----------------------------------------------

Benchmark:  https://zenodo.org/records/4624805
ENTSO-E:    Downloaded via ENTSO-E Transparency Platform API (script: download_entsoe.py)
