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
