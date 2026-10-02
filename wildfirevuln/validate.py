"""Fold logic for the leave-one-fire-out benchmark (see experiments/run_validation.py)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import product, taxonomy as T
from .model import CATS, Design, VulnModel

KS = [10, 25, 50, 100, 200]
EPS = 1e-6

COMPLIANT = {"roof": {"tile", "metal", "asphalt"}, "eaves": {"enclosed", "none"},
             "vents": {"fine", "none"}, "siding": {"noncomb"}, "windows": {"multi"},
             "deck": {"none"}, "patio": {"none", "noncomb"}, "fence": {"none", "noncomb"}}


def checklist_score(df):
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for f, good in COMPLIANT.items():
        known = df[f] != T.UNKNOWN
        num += (known & df[f].isin(good)).to_numpy()
        den += known.to_numpy()
    return np.where(den > 0, num / np.maximum(den, 1), 0.5)


def draw_impute(df, ref, seed):
    """Fill 'unknown' with random draws from the known levels in ref, so an
    unknown field cannot be told apart from a known one."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in CATS:
        known = ref[c][ref[c] != T.UNKNOWN].value_counts(normalize=True)
        m = (out[c] == T.UNKNOWN).to_numpy()
        if m.any() and len(known):
            out.loc[m, c] = rng.choice(known.index.to_numpy(), m.sum(), p=known.to_numpy())
    return out


def gbm_fit_predict(tr, te, cats, y_tr):
    dz = Design(cats=cats, use_num=True).fit(tr)
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, l2_regularization=1.0,
                                       random_state=0)
    g.fit(dz.transform(tr), y_tr)
    return g.predict_proba(dz.transform(te))[:, 1]


def metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {"brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else np.nan,
            "share_err": float(abs(p.mean() - y.mean()))}


def run_fold(d, sp, ev):
    tr, te = d[d.event != ev], d[d.event == ev]
    y_tr, y_te = tr.burnt.to_numpy(float), te.burnt.to_numpy(float)
    preds = {"constant": np.full(len(te), y_tr.mean())}

    m = VulnModel(Design(cats=["struct"], use_num=False), event_effect=False).fit(tr)
    preds["struct_only"] = m.predict(te)

    lr = LogisticRegression().fit(checklist_score(tr)[:, None], y_tr)
    preds["checklist"] = lr.predict_proba(checklist_score(te)[:, None])[:, 1]

    m = VulnModel(Design(cats=CATS, use_num=False), event_effect=False).fit(tr)
    preds["survey_glm"] = m.predict(te)

    preds["gbm_leaky"] = gbm_fit_predict(tr, te, CATS, y_tr)
    seed = zlib.crc32(ev.encode())
    tr_d, te_d = draw_impute(tr, tr, seed), draw_impute(te, tr, seed + 1)
    preds["gbm_survey"] = gbm_fit_predict(tr_d, te_d, CATS, y_tr)

    trp, tep = product.apply(tr, sp), product.apply(te, sp)
    p_gbm = gbm_fit_predict(product.apply(tr_d, sp), product.apply(te_d, sp), list(sp), y_tr)
    preds["gbm_graded"] = p_gbm
    wf = VulnModel(Design(cats=list(sp), use_num=True)).fit(trp)
    p_wf = wf.predict(tep)
    preds["wildfirevuln"] = p_wf

    # Event-level 80% predictive interval for the destroyed share.
    eta = wf.linpred(tep)
    z = 1.2815516 * np.sqrt(wf.tau2)
    share_lo, share_hi = expit(eta - z).mean(), expit(eta + z).mean()

    rows = [{"event": ev, "model": k, "n": len(te), "rate": y_te.mean(),
             **metrics(y_te, v)} for k, v in preds.items()]
    interval = {"event": ev, "rate": float(y_te.mean()), "pred": float(p_wf.mean()),
                "lo80": float(share_lo), "hi80": float(share_hi), "n": len(te),
                "tau": float(np.sqrt(wf.tau2))}

    live = []
    order = np.argsort(te.entry_order.to_numpy(), kind="stable")
    for k in KS:
        if k >= len(te) - 50:
            continue
        obs, rest = order[:k], order[k:]
        u, v = wf.event_posterior(tep.iloc[obs])
        y_rest = y_te[rest]
        p_live = wf.predict(tep.iloc[rest], u_mean=u, u_var=v)
        # GBM given the same k inspections: shift its logit so its mean over
        # the first k matches the observed share (shrunk with a Jeffreys prior).
        target = (y_te[obs].sum() + 0.5) / (k + 1.0)
        lg = logit(np.clip(p_gbm, EPS, 1 - EPS))
        lo_, hi_ = -10.0, 10.0
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            if expit(lg[obs] + mid).mean() < target:
                lo_ = mid
            else:
                hi_ = mid
        p_gbm_live = expit(lg[rest] + (lo_ + hi_) / 2)
        for name, p in [("wildfirevuln_prior", p_wf[rest]), ("wildfirevuln_live", p_live),
                        ("gbm_graded_prior", p_gbm[rest]), ("gbm_graded_recentred", p_gbm_live)]:
            live.append({"event": ev, "k": k, "model": name, **metrics(y_rest, p)})

    calib = pd.DataFrame({"event": ev, "y": y_te, "wildfirevuln": p_wf,
                          "gbm_graded": p_gbm, "survey_glm": preds["survey_glm"],
                          "checklist": preds["checklist"]})
    return rows, interval, live, calib
