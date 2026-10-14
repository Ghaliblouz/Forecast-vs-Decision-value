# Electricity Price Forecast Quality vs. Decision Value

Code for the master's thesis (emlyon DSAIS). Revised on 14 October 2026 after the feedback of 13 October 2026: see the revision log below.

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace cross_market_decision.ipynb   # about 1.5 min
jupyter nbconvert --to notebook --execute --inplace draft_data.ipynb              # about 2 min
jupyter nbconvert --to notebook --execute --inplace lear_new_data.ipynb           # regret part, from saved forecasts
cd .. && python -m pytest tests/                                                    # checks the shared helpers
```

The notebooks read the benchmark forecasts from `model_outputs/benchmark_forecasts/` and write figures to `figures/`.

## Code

| File | Content |
|---|---|
| `notebooks/decision_utils.py` | Shared helpers: predictive distribution (thesis eq. 3.5), forward proxy (eq. 3.6), regret, hedging decisions, Diebold–Mariano test, calibration. Every notebook imports them. |
| `notebooks/cross_market_decision.ipynb` | Reference analysis, five markets: regret (Table 4.6), DM tests (Tables 4.5 and 4.7), what decides which model wins (tail bias, disagreement hours), rank correlation over 10 models (Table 4.8). |
| `notebooks/draft_data.ipynb` | Exploratory log: data validation, price distributions, error by decile, interval calibration (Table 4.3), Nord Pool regret (Table 4.4) and DM tests. |
| `notebooks/lear_new_data.ipynb` | ENTSO-E 2022–2023 extension with one model (AR-LEAR). `RETRAIN = False` loads the saved forecasts; `RETRAIN = True` needs `epftoolbox` and the raw ENTSO-E files. |
| `notebooks/train_models.ipynb` | LEAR, LightGBM and DNN retraining. Needs `epftoolbox`, the raw Zenodo files, LightGBM and PyTorch; not re-run in this revision. |
| `notebooks/entso_data_analysis.ipynb` | ENTSO-E 2019–2023 data exploration; needs the raw ENTSO-E files. |
| `tests/test_decision_utils.py` | Checks the vectorised helpers against an hour-by-hour implementation of the thesis definitions. |

## Method in one paragraph

At each hour the predictive distribution is the point forecast plus the residuals (actual − forecast) of the previous 90 days. The buyer hedges (z = 1) when the forward proxy (90-day mean of past prices × 1.05) is below M_λ = (1 − λ) E[p] + λ CVaR₁₀%[p]. Regret is the extra cost versus perfect foresight. DNN and LEAR are compared with a Diebold–Mariano test on the daily mean loss differential (HAC, 7 daily lags, two-sided).

## Key results (benchmark 2015–2018, π = 5 %)

| Market | MSE DM p | MAE DM p | First λ with DNN significantly lower regret | Regret gap LEAR − DNN at λ = 1 (p) | Rank correlation MAE vs regret, λ = 0 → 1 |
|---|---|---|---|---|---|
| NP  | 0.224 | 0.042  | none at 5 % | +0.038 (0.102) | 0.95 → 0.47 |
| PJM | 0.555 | 0.001  | 0.1 | +0.348 (0.001) | 0.85 → 0.90 |
| BE  | 0.958 | <0.001 | 0.2 | +0.211 (<0.001) | 0.82 → 0.84 |
| FR  | 0.278 | 0.104  | 0.7 | +0.172 (<0.001) | 0.94 → 0.92 |
| DE  | 0.004 | 0.003  | 0.2 | +0.368 (<0.001) | 0.95 → 0.76 |

- At λ = 0 no market shows a significant regret difference, and the MAE and regret rankings of the 10 models agree closely.
- LEAR is never significantly better than the DNN. The onset, size and shape of the regret gap are market-specific.
- In the hours where the two models disagree at λ = 1, the oracle sides with the model that does not hedge in 96–100 % of cases; the DNN triggers fewer unnecessary hedges in all five markets.
- The predictive distribution is close to calibrated overall (80 % interval: 77.5–78.9 % coverage), but when a high price is forecast, the price exceeds the predicted 90th percentile in 13.8–26.8 % of hours instead of 10 %.

## Revision log — 14 October 2026

| Issue | Change | Where |
|---|---|---|
| Predictive distribution used p̂ − p, the mirror of eq. 3.5 (not in the feedback) | Residuals p − p̂, one implementation for all notebooks | `decision_utils.predictive_stats` |
| 6.1 One-sided DM test could not show LEAR better | Two-sided test | `decision_utils.dm_test` |
| 6.2 Hourly test with 12 lags ignored dependence within a day | Test on daily means, HAC 7 daily lags; applied to MSE, MAE and regret | `decision_utils.dm_test`, `cross_market` Section 2 |
| 3.2 / Table 4.5: MAE DM test had no code | MAE and MSE tests added | `cross_market` Section 2, `draft_data` Phase 5 |
| 4 Gap-factor ratios (28×, 6×) | Removed; gaps reported with their p-values | `cross_market` Section 1, `draft_data` Phase 4+5 |
| 5 Deciles grouped by realised price (selection effect) | Bias and calibration grouped by forecast decile | `decision_utils.tail_calibration`, `cross_market` Section 4, `draft_data` mechanism checks |
| 2 Disagreement breakdown on NP and PJM only, at π = 0 % | All five markets at π = 5 % | `cross_market` Section 4 |
| 7 ENTSO-E forward proxy included the current price | Past prices only, as in the benchmark | `lear_new_data` regret cell |
| Notebooks used absolute paths outside the repository | Paths relative to the repository | `decision_utils.ROOT` |
| Interpretation cells described results that no longer hold | Rewritten from the new outputs; earlier claims kept with a correction note in `draft_data` | all notebooks |

Not changed in code: feedback point 1.1 (reference [4]) and 1.2 (§6.1 has no notebook) concern the thesis text.

## Figures

| File | Content |
|---|---|
| `cross_market_centrepiece.png` | Regret gap and DM t-statistic vs λ, five markets |
| `cross_market_sensitivity.png` | Regret gap vs λ at π = 0 %, 5 %, 10 % |
| `tail_bias_by_forecast_decile.png` | Signed error by forecast decile, five markets |
| `disagreement_breakdown.png` | Disagreement hours at λ = 1, five markets |
| `rank_correlation.png` | Spearman ρ between MAE and regret rankings, 10 models |
| `regret_vs_lambda_NP.png`, `sensitivity_premium_NP.png` | Nord Pool regret vs λ and premium sensitivity |
| `cross_market_regret_dm.png` | Regret gap and DM t-statistic (exploratory notebook) |
| `price_distributions.png`, `seasonal_structure.png` | Benchmark price data |
| `error_by_decile_NP.png`, `error_by_decile_all.png` | Error by realised-price decile (descriptive; selection-biased for tail claims) |
| `bias_by_forecast_decile_NP_PJM.png`, `disagreement_NP_PJM.png`, `premium_sensitivity_NP_PJM.png` | NP and PJM detail |
| `new_period_regret_comparison.png` | ENTSO-E 2022–2023 vs 2016–2018 regret, one model per period |

## Model outputs

- `model_outputs/benchmark_forecasts/`: published DNN and LEAR forecasts from Lago et al. (2021), 5 CSVs, 17,472 hours each.
- `model_outputs/retrained_forecasts/`: retrained LEAR per window and ensemble on the benchmark (`{market}_LEAR_*_zenodo.csv`), AR-LEAR on ENTSO-E 2022–2023 (`{market}_ARLEAR_entso_2022_2023.csv`), LightGBM on NP.

## Data sources (raw data not included)

- Benchmark: https://zenodo.org/records/4624805
- ENTSO-E: day-ahead prices from the ENTSO-E Transparency Platform, expected in `data/raw/entso_e/` for retraining.
