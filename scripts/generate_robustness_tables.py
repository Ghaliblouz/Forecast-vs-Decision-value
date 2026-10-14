"""
generate_robustness_tables.py — tables for the robustness checks of Chapter 6 (Limitations),
computed with the same helpers as the notebooks (notebooks/decision_utils.py).

Run from the repository root:
    python scripts/generate_robustness_tables.py                 # writes to reports/generated_tables/
    python scripts/generate_robustness_tables.py --out <folder>

Tables (about 10 minutes, mostly the block-hedging linear programs):
    tab_6_1_block_hedging.tex     \\label{tab:block_hedging}     §6.1 partial hedging with two block products
    tab_6_4_buyer_criterion.tex   \\label{tab:buyer_criterion}   §6.4 models ranked on the buyer's own objective
    tab_6_6_holm.tex              \\label{tab:holm}              §6.6 Holm correction of the regret DM tests

Same table style as scripts/generate_tables.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "scripts"))
import decision_utils as du  # noqa: E402
from generate_tables import num, pval, table  # noqa: E402

SHOW = [0.0, 0.5, 1.0]


def compute():
    res = {}
    for m in du.MARKETS:
        df = du.load_market(m)
        f = du.forward_proxy(df["Real price"])
        reg = {k: du.compute_regret(df, f"{k} Ensemble", f) for k in ("DNN", "LEAR")}
        cost = {k: du.hedging_costs(df, f"{k} Ensemble", f) for k in ("DNN", "LEAR")}
        dm, buyer = [], []
        for lam in du.LAMBDAS:
            t, p = du.dm_test(reg["LEAR"][f"regret_λ={lam:.1f}"], reg["DNN"][f"regret_λ={lam:.1f}"],
                              reg["DNN"].index)
            dm.append({"lambda": lam, "t": t, "p": p})
            d, lo, hi, pb = du.block_bootstrap_diff(cost["LEAR"][f"cost_λ={lam:.1f}"],
                                                    cost["DNN"][f"cost_λ={lam:.1f}"],
                                                    cost["DNN"].index, lam)
            buyer.append({"lambda": lam, "diff": d, "lo": lo, "hi": hi, "p": pb,
                          "rho_dnn": du.buyer_objective(cost["DNN"][f"cost_λ={lam:.1f}"].values, lam)})
        dm = pd.DataFrame(dm).set_index("lambda")
        dm["p_holm11"] = du.holm(dm["p"])
        block = {k: du.block_hedging(df, f"{k} Ensemble") for k in ("DNN", "LEAR")}
        res[m] = {"dm": dm, "buyer": pd.DataFrame(buyer).set_index("lambda"), "reg": reg, "block": block}
        print(f"  {m}: computed", flush=True)
    # Holm over all 5 markets x 11 λ = 55 tests
    allp = np.concatenate([res[m]["dm"]["p"].values for m in du.MARKETS])
    adj = du.holm(allp).reshape(len(du.MARKETS), -1)
    for i, m in enumerate(du.MARKETS):
        res[m]["dm"]["p_holm55"] = adj[i]
    return res


def first_lam(dm, pcol):
    dnn, lear = du.first_significant(dm, pcol=pcol)
    parts = ([f"{dnn:.1f}"] if dnn is not None else []) + ([f"{lear:.1f} (LEAR)"] if lear is not None else [])
    return " / ".join(parts) or "none"


def n_sig(dm, pcol):
    return int((dm[pcol] < 0.05).sum())


def t66(res):
    rows = []
    for m, r in res.items():
        dm = r["dm"]
        rows.append(f"{m} & {first_lam(dm, 'p')} & {n_sig(dm, 'p')} & {first_lam(dm, 'p_holm11')} & "
                    f"{n_sig(dm, 'p_holm11')} & {first_lam(dm, 'p_holm55')} & {n_sig(dm, 'p_holm55')} \\\\")
    return table("Regret Diebold--Mariano tests (LEAR vs.\\ DNN) with and without a Holm correction for "
                 "multiple testing. First sig.\\ $\\lambda$: smallest $\\lambda$ at which the DNN has a "
                 "significantly lower regret at 5\\%; \\# sig.: number of the 11 values of $\\lambda$ "
                 "that are significant. Holm is applied to the 11 tests of each market, and to all 55 "
                 "tests together.",
                 "lC{1.6cm}C{1.0cm}C{1.6cm}C{1.0cm}C{1.6cm}C{1.0cm}",
                 ["Market", "First sig.\\ $\\lambda$, unadjusted", "\\# sig.", "First sig.\\ $\\lambda$, Holm (11)",
                  "\\# sig.", "First sig.\\ $\\lambda$, Holm (55)", "\\# sig."],
                 rows, "tab:holm")


def t64(res):
    rows = []
    for m, r in res.items():
        cells = []
        for lam in SHOW:
            b = r["buyer"].loc[lam]
            if b["diff"] == b["lo"] == b["hi"] == 0:     # identical costs in the tail (BE, λ = 1)
                cells.append("0 (identical)")
                continue
            cell = f"{num(b['diff'], 3, sign=True)}\\newline [{num(b['lo'], 3)}, {num(b['hi'], 3)}]"
            cells.append(f"\\tsig{{{cell}}}" if b["p"] < 0.05 else cell)
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    return table("The buyer's own criterion: difference $\\rho_\\lambda(\\text{LEAR}) - \\rho_\\lambda"
                 "(\\text{DNN})$ of the objective~\\eqref{eq:objective} evaluated on the costs actually "
                 "paid over the test period, $(1-\\lambda)$ mean cost $+\\ \\lambda$ CVaR$_{10\\%}$ of the "
                 "hourly cost (EUR/MWh; PJM in USD/MWh). Positive = the DNN gives the buyer a lower "
                 "objective. In brackets: 95\\% moving-block bootstrap interval (blocks of 7 days, 2{,}000 "
                 "draws); \\tsig{bold} = interval excludes zero. 0 (identical): both models pay "
                 "exactly the same cost in the 10\\% most expensive hours.",
                 "lC{3.6cm}C{3.6cm}C{3.6cm}",
                 ["Market"] + [f"$\\lambda = {lam:.1f}$" for lam in SHOW],
                 rows, "tab:buyer_criterion")


def t61(res):
    rows = []
    for m, r in res.items():
        b = r["block"]
        interior = max(b[k].groupby("lam")["interior"].mean().max() for k in b)
        cells = []
        for lam in (0.5, 1.0):
            col = f"regret_λ={lam:.1f}"
            hr = r["reg"]
            gh = hr["LEAR"][col].mean() - hr["DNN"][col].mean()
            ph = du.dm_test(hr["LEAR"][col], hr["DNN"][col], hr["DNN"].index)[1]
            a = b["LEAR"][b["LEAR"]["lam"] == lam].set_index("day")
            d = b["DNN"][b["DNN"]["lam"] == lam].set_index("day")
            gb = a["regret_bin"].mean() - d["regret_bin"].mean()
            pb = du.dm_test(a["regret_bin"], d["regret_bin"], d.index)[1]
            gc = a["regret_cont"].mean() - d["regret_cont"].mean()
            pc = du.dm_test(a["regret_cont"], d["regret_cont"], d.index)[1]
            cells += [f"{num(gh, 3, sign=True)} ({pval(ph)})", f"{num(gb, 3, sign=True)} ({pval(pb)})",
                      f"{num(gc, 3, sign=True)} ({pval(pc)})"]
        rows.append(f"{m} & {100 * interior:.1f} & " + " & ".join(cells) + " \\\\")
    return table("Partial hedging with two daily block products (peak 08--19\\,h, off-peak), solved with "
                 "the Rockafellar--Uryasev linear program. Interior: largest share of days, over the 11 "
                 "values of $\\lambda$ and the two models, with $0 < z_b < 1$ for a block. Regret gap "
                 "LEAR $-$ DNN (EUR/MWh; PJM in USD/MWh) and daily Diebold--Mariano $p$-value under the "
                 "hourly binary rule of Chapter~\\ref{ch:findings}, the block rule restricted to "
                 "$z_b \\in \\{0, 1\\}$, and the block rule with $z_b \\in [0, 1]$.",
                 "lC{1.3cm}C{1.55cm}C{1.55cm}C{1.55cm}C{1.55cm}C{1.55cm}C{1.55cm}",
                 ["Market", "Interior (\\%)", "$\\lambda{=}0.5$ hourly", "$\\lambda{=}0.5$ block, binary",
                  "$\\lambda{=}0.5$ block, partial", "$\\lambda{=}1$ hourly", "$\\lambda{=}1$ block, binary",
                  "$\\lambda{=}1$ block, partial"],
                 rows, "tab:block_hedging")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "generated_tables"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    print("Computing (about 10 minutes)...")
    res = compute()
    for name, fn in {"tab_6_1_block_hedging.tex": t61, "tab_6_4_buyer_criterion.tex": t64,
                     "tab_6_6_holm.tex": t66}.items():
        (out / name).write_text(fn(res), encoding="utf-8")
        print(f"  wrote {name}")
    # plain-text summary for checking the numbers quoted in the text
    for m, r in res.items():
        print(f"\n{m}\n", pd.concat([r["dm"].round(4), r["buyer"].round(4).add_prefix("buyer_")], axis=1).to_string())
        for k, b in r["block"].items():
            print(f"  block {k}:", b.groupby("lam")[["interior", "regret_bin", "regret_cont"]].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
