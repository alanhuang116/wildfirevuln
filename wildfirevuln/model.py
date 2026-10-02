"""WildfireVuln model: P(structure destroyed | fire reaches its neighbourhood).

    logit p_i = mu + x_i' beta + u_e(i),     u_e ~ N(0, tau^2)

x carries the harmonised building features (one-hot, reference level dropped)
and neighbourhood spacing. 'unknown' is never a parameter: an unknown field is
filled with the training frequencies of its known levels, so the model gives
it the population-average contribution. In DINS 'unknown' is far more common
on destroyed structures (an inspector cannot see the eaves of a burnt house),
so a dedicated unknown level would leak the outcome.

tau is estimated by Laplace-EM. For a fire the model has not seen, the
prediction integrates over u (prior predictive). When early inspections of a
live fire arrive, u is updated from them (event calibration).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import taxonomy as T

CATS = ["struct", "era"] + list(T.FIELDS)
REF = {f: v[2] for f, v in T.FIELDS.items()}
REF.update({"struct": "sfr_1", "era": "pre1990"})
NUM = ["log_n30", "log_n100", "log_nn", "log_n30_sq", "log_n100_sq", "log_nn_sq",
       "isolated", "n30_x_n100"]
GH_X, GH_W = np.polynomial.hermite.hermgauss(24)


def numeric(df):
    """Spacing terms. Quadratic in the logs: the destroyed share rises with
    density and then flattens, which a linear term cannot follow."""
    a = np.log1p(df["n30"].to_numpy(float))
    b = np.log1p(df["n100"].to_numpy(float))
    c = np.log(df["nn_m"].to_numpy(float) + 1.0)
    return {"log_n30": a, "log_n100": b, "log_nn": c,
            "log_n30_sq": a * a, "log_n100_sq": b * b, "log_nn_sq": c * c,
            "isolated": (df["n30"].to_numpy(float) == 0).astype(float),
            "n30_x_n100": a * b}


class Design:
    """Column layout plus the statistics needed to rebuild it in JavaScript."""

    def __init__(self, cats=CATS, use_num=True, extra=(), unknown="impute", interact=False):
        self.cats = list(cats)
        # interact: building-feature and spacing columns get a second copy
        # that is non-zero only for commercial structures, so commercial
        # buildings have their own feature effects while sharing fire effects.
        self.interact = interact
        self.use_num = use_num
        self.extra = list(extra)        # already-numeric columns, standardised
        self.unknown = unknown          # "impute" (product) or "indicator" (analysis)

    def fit(self, df):
        self.levels, self.freq = {}, {}
        for c in self.cats:
            known = df[c][df[c] != T.UNKNOWN]
            lv = sorted(v for v in known.unique() if v != REF[c])
            if self.unknown == "indicator" and (df[c] == T.UNKNOWN).any():
                lv.append(T.UNKNOWN)
            self.levels[c] = lv
            vc = known.value_counts(normalize=True)
            self.freq[c] = {v: float(vc.get(v, 0.0)) for v in lv}
        self.mean, self.sd = {}, {}
        if self.use_num:
            num = numeric(df)
            for k in NUM:
                self.mean[k] = float(num[k].mean())
                self.sd[k] = float(num[k].std())
        for k in self.extra:
            self.mean[k] = float(df[k].mean())
            self.sd[k] = float(df[k].std())
        self.columns = [f"{c}={v}" for c in self.cats for v in self.levels[c]]
        if self.use_num:
            self.columns += NUM
        self.columns += self.extra
        self.base = list(self.columns)
        if self.interact:
            self.feat_idx = [i for i, c in enumerate(self.base) if not c.startswith("struct=")]
            self.columns += ["com:" + self.base[i] for i in self.feat_idx]
        return self

    def transform(self, df):
        cols = []
        for c in self.cats:
            vals = df[c].to_numpy()
            unk = vals == T.UNKNOWN
            for v in self.levels[c]:
                col = (vals == v).astype(float)
                if v != T.UNKNOWN:
                    col[unk] = self.freq[c][v] if self.unknown == "impute" else 0.0
                cols.append(col)
        if self.use_num:
            num = numeric(df)
            for k in NUM:
                cols.append((num[k] - self.mean[k]) / self.sd[k])
        for k in self.extra:
            cols.append((df[k].to_numpy(float) - self.mean[k]) / self.sd[k])
        X = np.column_stack(cols) if cols else np.zeros((len(df), 0))
        if self.interact:
            com = df["struct"].isin(T.COMMERCIAL_STRUCT).to_numpy(float)
            X = np.column_stack([X, X[:, self.feat_idx] * com[:, None]])
        return X

    def to_json(self):
        return {"cats": self.cats, "levels": self.levels, "freq": self.freq,
                "ref": {c: REF[c] for c in self.cats}, "num": NUM if self.use_num else [],
                "mean": self.mean, "sd": self.sd, "columns": self.columns,
                "interact": self.interact, "com_struct": T.COMMERCIAL_STRUCT,
                "feat_idx": getattr(self, "feat_idx", [])}


def _nll(theta, X, y, ev, n_ev, tau2, ridge):
    k = X.shape[1]
    mu, beta, u = theta[0], theta[1:k + 1], theta[k + 1:]
    eta = mu + X @ beta + (u[ev] if n_ev else 0.0)
    p = expit(eta)
    ll = np.sum(np.logaddexp(0.0, eta) - y * eta)
    r = p - y
    g = np.empty_like(theta)
    g[0] = r.sum()
    g[1:k + 1] = X.T @ r + ridge * beta
    ll += 0.5 * ridge * beta @ beta
    if n_ev:
        g[k + 1:] = np.bincount(ev, weights=r, minlength=n_ev) + u / tau2
        ll += 0.5 * u @ u / tau2
    return ll, g


class VulnModel:
    def __init__(self, design: Design, event_effect=True, ridge=1.0, em_iter=12):
        self.design = design
        self.event_effect = event_effect
        self.ridge = ridge
        self.em_iter = em_iter

    def fit(self, df, y=None):
        y = df["burnt"].to_numpy(float) if y is None else np.asarray(y, float)
        self.design.fit(df)
        X = self.design.transform(df)
        k = X.shape[1]
        if self.event_effect:
            codes, ev = np.unique(df["event"].to_numpy(), return_inverse=True)
            n_ev = len(codes)
        else:
            codes, ev, n_ev = np.array([]), np.zeros(len(df), int), 0
        theta = np.zeros(1 + k + n_ev)
        theta[0] = np.log(y.mean() / (1 - y.mean()))
        tau2 = 1.0
        for _ in range(self.em_iter if n_ev else 1):
            res = minimize(_nll, theta, args=(X, y, ev, n_ev, tau2, self.ridge),
                           jac=True, method="L-BFGS-B", options={"maxiter": 2000})
            theta = res.x
            if not n_ev:
                break
            u = theta[k + 1:]
            p = expit(theta[0] + X @ theta[1:k + 1] + u[ev])
            h = np.bincount(ev, weights=p * (1 - p), minlength=n_ev) + 1.0 / tau2
            tau2_new = float(np.mean(u ** 2 + 1.0 / h))
            if abs(tau2_new - tau2) < 1e-4:
                tau2 = tau2_new
                break
            tau2 = tau2_new
        self.mu, self.beta = float(theta[0]), theta[1:k + 1]
        self.tau2 = tau2 if n_ev else 0.0
        self.events = {str(c): float(v) for c, v in zip(codes, theta[k + 1:])}
        # Approximate covariance of (mu, beta) for parameter uncertainty.
        p = expit(theta[0] + X @ self.beta + (theta[k + 1:][ev] if n_ev else 0.0))
        Xa = np.column_stack([np.ones(len(X)), X])
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa
        H[1:, 1:] += self.ridge * np.eye(k)
        self.cov = np.linalg.pinv(H)
        return self

    def linpred(self, df):
        return self.mu + self.design.transform(df) @ self.beta

    def predict(self, df, u_mean=0.0, u_var=None):
        """Probability integrated over the event effect N(u_mean, u_var)."""
        eta = self.linpred(df)
        var = self.tau2 if u_var is None else u_var
        if var <= 0:
            return expit(eta + u_mean)
        s = np.sqrt(2.0 * var)
        return expit(eta[:, None] + u_mean + s * GH_X[None, :]) @ GH_W / np.sqrt(np.pi)

    def event_posterior(self, df_obs):
        """Laplace posterior of a new fire's event effect from early inspections."""
        eta = self.linpred(df_obs)
        y = df_obs["burnt"].to_numpy(float)
        u = 0.0
        for _ in range(50):
            p = expit(eta + u)
            g = np.sum(y - p) - u / self.tau2
            h = np.sum(p * (1 - p)) + 1.0 / self.tau2
            step = g / h
            u += step
            if abs(step) < 1e-8:
                break
        return u, 1.0 / h

    def to_json(self):
        return {"mu": self.mu, "beta": self.beta.tolist(), "tau2": self.tau2,
                "design": self.design.to_json(), "events": self.events,
                "se": np.sqrt(np.diag(self.cov)).tolist()}


def dump(model, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model.to_json(), fh, indent=1)
