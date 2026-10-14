"""Shared decision-analysis helpers used by every notebook.

Single source of truth for the predictive distribution, the hedging decision,
the regret and the Diebold-Mariano test, so that a fix applies everywhere.

Predictive distribution (thesis, eq. 3.5)
-----------------------------------------
At hour t the buyer only has a point forecast p̂_t. The predictive distribution
is the empirical distribution of

    p̂_t + (p_s - p̂_s),    s = t - W, ..., t - 1

i.e. today's forecast shifted by the last W hourly *residuals*
(actual - forecast). A residual is positive when the price came out above the
forecast, so price spikes populate the upper tail, which is what CVaR reads.

Note: the notebooks up to commit 7b51259 used (p̂_s - p_s) instead, which
mirrors the distribution and moves price spikes into the lower tail.

Decision and regret (thesis, Proposition 1 and eq. 4.1)
-------------------------------------------------------
    M_λ  = (1 - λ) E[p] + λ CVaR_α[p]
    z*   = 1 if f_t < M_λ else 0          (hedge at the forward price)
    z_or = 1 if f_t < p_t else 0          (perfect-foresight decision)
    R_t  = cost(z*) - cost(z_or) >= 0,   cost(z) = z f_t + (1 - z) p_t

Diebold-Mariano test
--------------------
The 24 hourly prices of a day are forecast together the day before, so hourly
losses within a day are strongly dependent. The test is run on the daily mean
of the loss differential, with HAC standard errors (7 daily lags), and is
two-sided so that either model can be found significantly better
(Lago et al., 2021, recommend testing at the daily level).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 90 * 24                                  # 90 days of hourly residuals
ALPHA = 0.10                                      # CVaR tail: worst 10 %
PREMIUM = 0.05                                    # forward = rolling mean x (1 + premium)
LAMBDAS = np.round(np.arange(0, 1.1, 0.1), 1)     # λ in {0, 0.1, ..., 1}
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]
DM_LAGS = 7                                       # HAC lags for the daily DM test (one week)


# ── Paths ───────────────────────────────────────────────────────────────────
def repo_root():
    """Repository root, whether the notebook runs from notebooks/ or from the root."""
    here = Path.cwd().resolve()
    return here.parent if here.name == "notebooks" else here


ROOT = repo_root()
DATA_DIR = ROOT / "model_outputs" / "benchmark_forecasts"
RETRAINED_DIR = ROOT / "model_outputs" / "retrained_forecasts"
FIG_DIR = ROOT / "figures"
MARKET_FILES = {m: DATA_DIR / f"Forecasts_{m}_DNN_LEAR_ensembles.csv" for m in MARKETS}


def load_market(market):
    """Benchmark forecasts (Lago et al., 2021) for one market, hourly DatetimeIndex."""
    return pd.read_csv(MARKET_FILES[market], index_col=0, parse_dates=True)


# ── Forward price proxy (thesis, eq. 3.6) ───────────────────────────────────
def forward_proxy(prices, window=WINDOW, premium=PREMIUM):
    """Rolling mean of the *past* `window` realised prices, times (1 + premium).

    shift(1) keeps the current hour's price out of its own forward price.
    """
    return prices.shift(1).rolling(window).mean() * (1 + premium)


# ── Predictive distribution ─────────────────────────────────────────────────
def predictive_stats(forecast, actual, window=WINDOW, alpha=ALPHA, quantiles=(), chunk=1000):
    """E[p], CVaR_alpha[p] and optional percentiles of the predictive distribution.

    Returns a dict of arrays aligned with hours t = window, ..., N-1:
      'mean', 'cvar', and one entry per requested percentile (e.g. 10, 90).
    Identical to building `pred_dist = forecast[t] + residuals[t-window:t]` hour
    by hour, but vectorised in chunks.
    """
    f = np.asarray(forecast, dtype=float)
    resid = np.asarray(actual, dtype=float) - f              # p - p̂  (eq. 3.5)
    win = sliding_window_view(resid, window)[:-1]            # row i -> resid[i : i + window]
    n = win.shape[0]
    out = {"mean": np.empty(n), "cvar": np.empty(n)}
    for q in quantiles:
        out[q] = np.empty(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        for s in range(0, n, chunk):
            pred = f[window + s: window + s + chunk, None] + win[s: s + chunk]
            thr = np.percentile(pred, (1 - alpha) * 100, axis=1)
            tail = pred >= thr[:, None]
            out["mean"][s: s + chunk] = pred.mean(axis=1)
            out["cvar"][s: s + chunk] = np.where(tail, pred, 0.0).sum(axis=1) / tail.sum(axis=1)
            for q in quantiles:
                out[q][s: s + chunk] = np.percentile(pred, q, axis=1)
    return out


# ── Decision and regret ─────────────────────────────────────────────────────
def compute_regret(df, model_col, f, window=WINDOW, alpha=ALPHA, lambdas=LAMBDAS):
    """Hourly regret vs perfect foresight, one column per λ ('regret_λ=0.0', ...).

    df must contain 'Real price' and `model_col`; f is the forward price series.
    Hours before `window` and hours with no forward price are dropped.
    """
    stats = predictive_stats(df[model_col].values, df["Real price"].values, window, alpha)
    ft = np.asarray(f, dtype=float)[window:]
    pt = df["Real price"].values[window:]
    keep = ~np.isnan(ft)
    cost_oracle = np.where(ft < pt, ft, pt)
    cols = {}
    for lam in lambdas:
        m_lam = (1 - lam) * stats["mean"] + lam * stats["cvar"]
        cost_star = np.where(ft < m_lam, ft, pt)
        cols[f"regret_λ={lam:.1f}"] = (cost_star - cost_oracle)[keep]
    out = pd.DataFrame(cols, index=df.index[window:][keep])
    out.index.name = "hour"
    return out


def decisions(df, model_col, f, lam, window=WINDOW, alpha=ALPHA):
    """Hedging decision z* of the model and of the oracle at one λ."""
    stats = predictive_stats(df[model_col].values, df["Real price"].values, window, alpha)
    ft = np.asarray(f, dtype=float)[window:]
    pt = df["Real price"].values[window:]
    keep = ~np.isnan(ft)
    m_lam = (1 - lam) * stats["mean"] + lam * stats["cvar"]
    out = pd.DataFrame({"z_star": (ft < m_lam).astype(int), "z_oracle": (ft < pt).astype(int)},
                       index=df.index[window:])
    return out[keep]


def disagreement(dec_a, dec_b):
    """Hours where only A hedges / only B hedges, and how often the oracle says 'don't hedge'."""
    only_a = (dec_a["z_star"] == 1) & (dec_b["z_star"] == 0)
    only_b = (dec_a["z_star"] == 0) & (dec_b["z_star"] == 1)
    no_hedge = dec_a["z_oracle"] == 0
    return {
        "only_A_hedges": int(only_a.sum()),
        "only_A_oracle_no_hedge": float(no_hedge[only_a].mean()) if only_a.any() else float("nan"),
        "only_B_hedges": int(only_b.sum()),
        "only_B_oracle_no_hedge": float(no_hedge[only_b].mean()) if only_b.any() else float("nan"),
    }


# ── Diebold-Mariano test ────────────────────────────────────────────────────
def dm_test(loss_a, loss_b, index, lags=DM_LAGS):
    """Two-sided DM test on the daily mean loss differential d = loss_a - loss_b.

    Returns (t, p). t > 0 means model B has the lower loss (B better);
    t < 0 means model A is better.
    """
    d = pd.Series(np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float),
                  index=pd.DatetimeIndex(index))
    d = d.groupby(d.index.normalize()).mean().dropna().values
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def first_significant(dm_df, level=0.05):
    """First λ where B (t > 0) or A (t < 0) is significantly better. dm_df: index λ, columns t, p."""
    b = dm_df[(dm_df["p"] < level) & (dm_df["t"] > 0)].index
    a = dm_df[(dm_df["p"] < level) & (dm_df["t"] < 0)].index
    return (float(b[0]) if len(b) else None, float(a[0]) if len(a) else None)


# ── Calibration ─────────────────────────────────────────────────────────────
def tail_calibration(df, model_col, window=WINDOW, top=0.9):
    """Calibration of the 80 % interval, overall and in the top decile.

    The top decile is defined two ways:
      - by FORECAST (what the buyer sees when deciding): free of selection effect;
      - by REALISED price: selects the hours with the largest positive shocks, so
        even a perfect forecast of the conditional mean looks biased there.
    Deciles are computed on the evaluation hours (t >= window).
    """
    fc = df[model_col].values
    y_all = df["Real price"].values
    s = predictive_stats(fc, y_all, window, quantiles=(10, 90))
    y, f = y_all[window:], fc[window:]
    lo, hi = s[10], s[90]
    inside = (y >= lo) & (y <= hi)
    above = y > hi
    top_fc = f >= np.quantile(f, top)
    top_real = y >= np.quantile(y, top)
    return {
        "cov80_all": inside.mean(),
        "cov80_top_forecast": inside[top_fc].mean(),
        "cov80_top_realised": inside[top_real].mean(),
        "exceed_q90_all": above.mean(),
        "exceed_q90_top_forecast": above[top_fc].mean(),
        "exceed_q90_top_realised": above[top_real].mean(),
        "bias_top_forecast": (f - y)[top_fc].mean(),
        "bias_top_realised": (f - y)[top_real].mean(),
    }
