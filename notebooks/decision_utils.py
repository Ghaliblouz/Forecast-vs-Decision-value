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

Block hedging with partial cover (thesis §6.1)
-----------------------------------------------
Two daily products (peak 08-19 h, off-peak 00-07 and 20-23 h), each bought in a
share z_b in [0, 1]. Because the CVaR of a sum is not the sum of the CVaRs, the
optimum can be interior; it is found with the Rockafellar-Uryasev linear program.
See block_scenarios and solve_block_hedge below.

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


def first_significant(dm_df, level=0.05, pcol="p"):
    """First λ where B (t > 0) or A (t < 0) is significantly better. dm_df: index λ, columns t, p.

    pcol selects the p-value column, e.g. a Holm-adjusted one.
    """
    b = dm_df[(dm_df[pcol] < level) & (dm_df["t"] > 0)].index
    a = dm_df[(dm_df[pcol] < level) & (dm_df["t"] < 0)].index
    return (float(b[0]) if len(b) else None, float(a[0]) if len(a) else None)


def holm(pvals):
    """Holm (1979) step-down adjusted p-values; controls the family-wise error rate.

    Sort the m p-values, multiply the k-th smallest by (m - k + 1), keep the sequence
    non-decreasing and cap at 1. A test is rejected at level a if its adjusted p < a.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


# ── The buyer's own criterion ───────────────────────────────────────────────
def hedging_costs(df, model_col, f, window=WINDOW, alpha=ALPHA, lambdas=LAMBDAS):
    """Hourly cost actually paid when following z* at each λ: z* f_t + (1 - z*) p_t.

    Same hours and column names ('cost_λ=0.0', ...) as compute_regret.
    """
    stats = predictive_stats(df[model_col].values, df["Real price"].values, window, alpha)
    ft = np.asarray(f, dtype=float)[window:]
    pt = df["Real price"].values[window:]
    keep = ~np.isnan(ft)
    cols = {}
    for lam in lambdas:
        m_lam = (1 - lam) * stats["mean"] + lam * stats["cvar"]
        cols[f"cost_λ={lam:.1f}"] = np.where(ft < m_lam, ft, pt)[keep]
    out = pd.DataFrame(cols, index=df.index[window:][keep])
    out.index.name = "hour"
    return out


def cvar(x, alpha=ALPHA, axis=-1):
    """Mean of the values at or above the (1 - alpha) percentile (same tail rule as predictive_stats)."""
    x = np.asarray(x, dtype=float)
    thr = np.expand_dims(np.percentile(x, (1 - alpha) * 100, axis=axis), axis)
    tail = x >= thr
    return np.where(tail, x, 0.0).sum(axis=axis) / tail.sum(axis=axis)


def buyer_objective(cost, lam, alpha=ALPHA, axis=-1):
    """The buyer's own criterion on the costs actually paid (thesis eq. objective):
    rho_λ = (1 - λ) mean(cost) + λ CVaR_α(cost), CVaR over the hours of the period."""
    cost = np.asarray(cost, dtype=float)
    return (1 - lam) * cost.mean(axis=axis) + lam * cvar(cost, alpha, axis)


def block_bootstrap_diff(cost_a, cost_b, index, lam, alpha=ALPHA, n_boot=2000, block_days=7, seed=0):
    """rho_λ(A) - rho_λ(B) with a paired moving-block bootstrap over days.

    CVaR is not an average of hourly losses, so the DM test does not apply. Whole days are
    resampled in blocks of `block_days` consecutive days (the same draw for both models),
    which keeps the within-day and day-to-day dependence.
    Returns (diff, ci_low, ci_high, p_two_sided); p is the percentile-bootstrap p-value.
    """
    idx = pd.DatetimeIndex(index)
    days = idx.normalize()
    a = pd.Series(np.asarray(cost_a, dtype=float), index=idx)
    b = pd.Series(np.asarray(cost_b, dtype=float), index=idx)
    full = a.groupby(days).size()
    full = full[full == full.max()].index                 # complete days only
    A = np.stack([a[days == d].values for d in full])     # (n_days, hours per day)
    B = np.stack([b[days == d].values for d in full])
    n = len(full)
    diff = buyer_objective(A.ravel(), lam, alpha) - buyer_objective(B.ravel(), lam, alpha)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_days))
    starts = rng.integers(0, n - block_days + 1, size=(n_boot, n_blocks))
    rows = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    boot = (buyer_objective(A[rows].reshape(n_boot, -1), lam, alpha)
            - buyer_objective(B[rows].reshape(n_boot, -1), lam, alpha))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return float(diff), float(lo), float(hi), float(p)


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


# ── Block hedging with partial cover (thesis §6.1) ──────────────────────────
PEAK_HOURS = tuple(range(8, 20))                  # 08:00-19:59, 12 hours; the rest is off-peak


def block_scenarios(df, model_col, window_days=WINDOW // 24, premium=PREMIUM, peak_hours=PEAK_HOURS):
    """Daily scenarios for the peak and off-peak block costs (1 MW in every hour).

    For day d the scenarios are today's block forecast plus the block residual
    (actual - forecast, summed over the block's hours) of each of the previous
    `window_days` days: the daily analogue of eq. 3.5, keeping the dependence
    between hours of the same day. The block forward price is the mean realised
    block price of the same past days times (1 + premium) (daily analogue of eq. 3.6).

    Returns a dict with arrays for days d = window_days, ..., D-1:
      'dates' (n,), 'f' (n, 2), 'scen' (n, window_days, 2), 'actual' (n, 2);
      column 0 = peak, column 1 = off-peak; all in EUR per block (sum of 12 hourly prices).
    """
    hours = df.index.hour.values
    if len(df) % 24 or not (hours.reshape(-1, 24) == np.arange(24)).all():
        raise ValueError("block_scenarios needs 24 consecutive hours (0-23) per day")
    fc = df[model_col].values.reshape(-1, 24)
    y = df["Real price"].values.reshape(-1, 24)
    peak = np.isin(np.arange(24), peak_hours)

    def blocks(a):
        return np.stack([a[:, peak].sum(axis=1), a[:, ~peak].sum(axis=1)], axis=1)   # (D, 2)

    fc_b, y_b = blocks(fc), blocks(y)
    res_b = y_b - fc_b                                        # block residuals, p - p̂
    w = window_days
    past_res = sliding_window_view(res_b, w, axis=0)[:-1]     # (D - w, 2, w): days d-w ... d-1
    past_y = sliding_window_view(y_b, w, axis=0)[:-1]
    scen = fc_b[w:, :, None] + past_res                       # (n, 2, w)
    return {
        "dates": df.index[::24][w:],
        "f": past_y.mean(axis=2) * (1 + premium),
        "scen": np.transpose(scen, (0, 2, 1)),                # (n, w, 2)
        "actual": y_b[w:],
    }


def ru_cvar(losses, alpha=ALPHA):
    """CVaR of equally likely scenarios as in Rockafellar and Uryasev (2000):
    the mean of the worst alpha * S scenarios (fractional weight on the boundary one)."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    k = alpha * len(x)
    full = int(np.floor(k + 1e-9))
    tail = x[:full].sum() + (k - full) * (x[full] if full < len(x) else 0.0)
    return tail / k


def solve_block_hedge(f, scen, lam, alpha=ALPHA):
    """Optimal shares z = (z_peak, z_offpeak) in [0, 1]^2 under rho_λ = (1-λ) E[cost] + λ CVaR_α[cost].

    cost_s(z) = sum_b z_b f_b + (1 - z_b) P_{s,b}. Rockafellar-Uryasev linear program:
      min (1-λ) mean_s cost_s(z) + λ (v + 1/(α S) sum_s u_s)
      s.t. u_s >= cost_s(z) - v,  u_s >= 0,  0 <= z_b <= 1.
    At λ = 0 the objective is linear in z, so z_b = 1{f_b < E[P_b]}.
    Returns (z, objective).
    """
    from scipy.optimize import linprog

    f = np.asarray(f, dtype=float)
    scen = np.asarray(scen, dtype=float)
    S = scen.shape[0]
    mean_p = scen.mean(axis=0)
    if lam == 0:
        z = (f < mean_p).astype(float)
        return z, float(z @ f + (1 - z) @ mean_p)
    c = np.concatenate([(1 - lam) * (f - mean_p), [lam], np.full(S, lam / (alpha * S))])
    A = np.zeros((S, 3 + S))
    A[:, 0:2] = f - scen
    A[:, 2] = -1.0
    A[:, 3:] = -np.eye(S)
    b = -scen.sum(axis=1)
    bounds = [(0, 1), (0, 1), (None, None)] + [(0, None)] * S
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return r.x[:2], float(r.fun + (1 - lam) * mean_p.sum())


def block_objective(z, f, scen, lam, alpha=ALPHA):
    """rho_λ of a given z on the scenarios (used for the binary corners and for checks)."""
    z = np.asarray(z, dtype=float)
    cost = scen @ (1 - z) + z @ f
    return (1 - lam) * cost.mean() + lam * ru_cvar(cost, alpha)


CORNERS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def block_hedging(df, model_col, lambdas=LAMBDAS, alpha=ALPHA, premium=PREMIUM, tol=1e-4):
    """Daily decisions and regret for the two block products, continuous and binary shares.

    Returns a long DataFrame (one row per day and λ) with columns
      lam, z_peak, z_offpeak, interior, regret_cont, regret_bin  (regret in EUR per MWh,
      i.e. daily regret / 24, comparable with the hourly analysis).
    regret = cost(z) - cost(oracle), oracle z_b = 1{f_b < realised block cost}.
    """
    s = block_scenarios(df, model_col, premium=premium)
    f, act = s["f"], s["actual"]
    z_or = (f < act).astype(float)
    cost_or = (z_or * f + (1 - z_or) * act).sum(axis=1)
    rows = []
    for i, day in enumerate(s["dates"]):
        sc = s["scen"][i]
        for lam in lambdas:
            z, _ = solve_block_hedge(f[i], sc, lam, alpha)
            z = np.clip(z, 0.0, 1.0)
            zb = CORNERS[np.argmin([block_objective(c, f[i], sc, lam, alpha) for c in CORNERS])]
            cost_c = (z * f[i] + (1 - z) * act[i]).sum()
            cost_b = (zb * f[i] + (1 - zb) * act[i]).sum()
            rows.append({
                "day": day, "lam": float(lam), "z_peak": z[0], "z_offpeak": z[1],
                "interior": bool(((z > tol) & (z < 1 - tol)).any()),
                "regret_cont": (cost_c - cost_or[i]) / 24, "regret_bin": (cost_b - cost_or[i]) / 24,
            })
    return pd.DataFrame(rows)
