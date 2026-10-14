"""Checks that the vectorised helpers match a plain hour-by-hour implementation.

Run from the repository root:  python -m pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import decision_utils as du  # noqa: E402

N_HOURS = du.WINDOW + 24 * 40   # 40 evaluation days keep the loop fast


def reference_regret(df, model_col, f, window, alpha, lambdas):
    """Hour-by-hour version of the thesis definition (eq. 3.5, Prop. 1, eq. 4.1)."""
    resid = df["Real price"] - df[model_col]            # p - p̂
    rows = []
    for t in range(window, len(df)):
        f_t, p_t = f.iloc[t], df["Real price"].iloc[t]
        if np.isnan(f_t):
            continue
        pred = df[model_col].iloc[t] + resid.iloc[t - window: t].values
        e_p = pred.mean()
        cvar = pred[pred >= np.percentile(pred, (1 - alpha) * 100)].mean()
        cost_or = f_t if f_t < p_t else p_t
        row = {"hour": df.index[t]}
        for lam in lambdas:
            m = (1 - lam) * e_p + lam * cvar
            row[f"regret_λ={lam:.1f}"] = (f_t if f_t < m else p_t) - cost_or
        rows.append(row)
    return pd.DataFrame(rows).set_index("hour")


def test_compute_regret_matches_reference():
    df = du.load_market("NP").iloc[:N_HOURS]
    f = du.forward_proxy(df["Real price"])
    for col in ["DNN Ensemble", "LEAR Ensemble"]:
        fast = du.compute_regret(df, col, f)
        slow = reference_regret(df, col, f, du.WINDOW, du.ALPHA, du.LAMBDAS)
        assert list(fast.index) == list(slow.index)
        np.testing.assert_allclose(fast.values, slow.values, atol=1e-9)


def test_predictive_distribution_uses_residual_sign():
    # A single positive spike in the past window must raise the upper tail, not the lower one.
    n = du.WINDOW + 1
    actual = np.full(n, 50.0)
    actual[10] = 500.0                                   # price spike the model missed
    forecast = np.full(n, 50.0)
    s = du.predictive_stats(forecast, actual, quantiles=(0.01, 99.99))
    assert s[99.99][0] > 400 and s[0.01][0] == 50.0


def test_dm_test_daily_two_sided():
    df = du.load_market("NP")
    p = df["Real price"]
    la = (df["LEAR Ensemble"] - p).abs()
    lb = (df["DNN Ensemble"] - p).abs()
    t, pval = du.dm_test(la, lb, df.index)
    d = (la - lb).groupby(df.index.date).mean().values
    ref = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
    assert np.isclose(t, ref.tvalues[0]) and np.isclose(pval, ref.pvalues[0])
    t_swapped, p_swapped = du.dm_test(lb, la, df.index)
    assert np.isclose(t_swapped, -t) and np.isclose(p_swapped, pval)


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.001, 0.04, 0.03, 0.2, 0.012, 0.6])
    np.testing.assert_allclose(du.holm(p), multipletests(p, method="holm")[1])


def test_buyer_objective_and_bootstrap():
    df = du.load_market("NP").iloc[:N_HOURS]
    f = du.forward_proxy(df["Real price"])
    cost = du.hedging_costs(df, "DNN Ensemble", f)
    reg = du.compute_regret(df, "DNN Ensemble", f)
    p = df["Real price"].loc[cost.index]
    oracle = np.minimum(f.loc[cost.index], p)
    # cost paid - oracle cost = regret
    np.testing.assert_allclose(cost["cost_λ=0.5"] - oracle, reg["regret_λ=0.5"], atol=1e-9)
    x = cost["cost_λ=1.0"].values
    tail = x[x >= np.percentile(x, 90)]
    assert np.isclose(du.buyer_objective(x, 1.0), tail.mean())
    assert np.isclose(du.buyer_objective(x, 0.0), x.mean())
    cost_b = du.hedging_costs(df, "LEAR Ensemble", f)
    d, lo, hi, pv = du.block_bootstrap_diff(cost_b["cost_λ=1.0"], x, cost.index, 1.0, n_boot=200)
    assert np.isclose(d, du.buyer_objective(cost_b["cost_λ=1.0"].values, 1.0) - tail.mean())
    assert lo <= hi and 0 <= pv <= 1


def test_block_hedge_lp_matches_grid():
    df = du.load_market("NP").iloc[: 24 * 100]
    s = du.block_scenarios(df, "DNN Ensemble")
    # scenarios use residual sign p - p̂: forecast + residual of a past day
    fc = df["DNN Ensemble"].values.reshape(-1, 24)
    y = df["Real price"].values.reshape(-1, 24)
    peak = np.isin(np.arange(24), du.PEAK_HOURS)
    assert np.isclose(s["scen"][0, 0, 0], fc[90, peak].sum() + (y[0, peak] - fc[0, peak]).sum())
    grid = np.linspace(0, 1, 41)
    for i in range(0, len(s["dates"]), 3):
        for lam in (0.0, 0.5, 1.0):
            z, obj = du.solve_block_hedge(s["f"][i], s["scen"][i], lam)
            assert np.isclose(obj, du.block_objective(z, s["f"][i], s["scen"][i], lam))
            best = min(du.block_objective([a, b], s["f"][i], s["scen"][i], lam) for a in grid for b in grid)
            assert obj <= best + 1e-6
