"""Fold logic for the commercial benchmark (see experiments/run_commercial.py)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import product, taxonomy as T
from .model import Design, VulnModel
from .validate import EPS, draw_impute, gbm_fit_predict, metrics

KS = [25, 50, 100, 200]


def recentre(lg, obs_idx_lg, target):
    a, b = -10.0, 10.0
    for _ in range(60):
        m = (a + b) / 2
        a, b = (m, b) if expit(obs_idx_lg + m).mean() < target else (a, m)
    return (a + b) / 2


def fold(d, sp, ev):
    tr = d[d.event != ev]
    te = d[d.event == ev]
    te_c, te_r = te[te.asset == "commercial"], te[te.asset == "residential"]
    tr_c, tr_r = tr[tr.asset == "commercial"], tr[tr.asset == "residential"]
    y = te_c.burnt.to_numpy(float)
    preds = {"constant": np.full(len(te_c), tr_c.burnt.mean())}

    res_sp = {f: [v for v in lv if v not in T.COMMERCIAL_STRUCT] for f, lv in sp.items()}
    m_res = VulnModel(Design(cats=list(res_sp))).fit(product.apply(tr_r, res_sp))
    as_home = product.apply(te_c.assign(struct="sfr_1"), res_sp)
    preds["home_curve"] = m_res.predict(as_home)

    m_com = VulnModel(Design(cats=list(sp))).fit(product.apply(tr_c, sp))
    preds["commercial_only"] = m_com.predict(product.apply(te_c, sp))

    seed = zlib.crc32(ev.encode())
    tr_d, te_d = draw_impute(tr, tr, seed), draw_impute(te, tr, seed + 1)
    p_gbm_all = gbm_fit_predict(product.apply(tr_d, sp), product.apply(te_d, sp), list(sp),
                                tr.burnt.to_numpy(float))
    is_c = (te.asset == "commercial").to_numpy()
    preds["gbm_joint"] = p_gbm_all[is_c]

    shared = VulnModel(Design(cats=list(sp))).fit(product.apply(tr, sp))
    preds["joint_shared_slopes"] = shared.predict(product.apply(te_c, sp))
    wf = VulnModel(product.design(sp)).fit(product.apply(tr, sp))
    tcp, trp = product.apply(te_c, sp), product.apply(te_r, sp)
    preds["wildfirevuln"] = wf.predict(tcp)

    eta = wf.linpred(tcp)
    z = 1.2815516 * np.sqrt(wf.tau2)
    interval = {"event": ev, "n": len(te_c), "rate": float(y.mean()),
                "pred": float(preds["wildfirevuln"].mean()),
                "lo80": float(expit(eta - z).mean()), "hi80": float(expit(eta + z).mean())}
    rows = [{"event": ev, "model": k, "n": len(te_c), "rate": float(y.mean()), **metrics(y, v)}
            for k, v in preds.items()]

    live = []
    order = np.argsort(te_r.entry_order.to_numpy(), kind="stable")
    y_r = te_r.burnt.to_numpy(float)
    lg_all = logit(np.clip(p_gbm_all, EPS, 1 - EPS))
    lg_c, lg_r = lg_all[is_c], lg_all[~is_c]
    for k in KS:
        if k > len(te_r):
            break
        obs = order[:k]
        u, v = wf.event_posterior(trp.iloc[obs])
        p_live = wf.predict(tcp, u_mean=u, u_var=v)
        shift = recentre(lg_r, lg_r[obs], (y_r[obs].sum() + 0.5) / (k + 1.0))
        p_g = expit(lg_c + shift)
        for name, p in [("wildfirevuln_prior", preds["wildfirevuln"]),
                        ("wildfirevuln_from_homes", p_live),
                        ("gbm_recentred_on_homes", p_g)]:
            live.append({"event": ev, "k": k, "model": name, **metrics(y, p)})
    return rows, interval, live


def boot_relativities(dp, cats, seed):
    rng = np.random.default_rng(seed)
    evs = dp.event.unique()
    pick = rng.choice(evs, size=len(evs), replace=True)
    parts = []
    for j, e in enumerate(pick):
        g = dp[dp.event == e].copy()
        g["event"] = f"{e}#{j}"
        parts.append(g)
    bd = pd.concat(parts, ignore_index=True)
    m = VulnModel(Design(cats=cats, interact=True), em_iter=6).fit(bd)
    return pd.Series(m.beta, index=m.design.columns)
