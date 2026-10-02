"""Property tests for the taxonomy, the design and the model."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wildfirevuln import taxonomy as T  # noqa: E402
from wildfirevuln.model import Design, VulnModel  # noqa: E402
from experiments.run_credits import credit  # noqa: E402


def test_harmonise():
    assert T.harmonise("Mesh Screen <= 1/8\"", T.VENTS) == "fine"
    assert T.harmonise("Ignition Resistant", T.SIDING) == "noncomb"
    assert T.harmonise("Attached Fence", T.VENTS) == T.UNKNOWN   # field spill-over
    assert T.harmonise(None, T.ROOF) == T.UNKNOWN
    assert T.era(2008) == "post2008" and T.era(0) == T.UNKNOWN


def synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "event": rng.choice([f"f{i}" for i in range(30)], n),
        "struct": rng.choice(["sfr_1", "sfr_2"], n), "era": "pre1990",
        "vents": rng.choice(["coarse", "fine"], n),
        "n30": rng.integers(0, 6, n), "n100": rng.integers(0, 40, n),
        "nn_m": rng.uniform(5, 80, n)})
    u = dict(zip(sorted(df.event.unique()), rng.normal(0, 1.2, 30)))
    eta = -0.2 + np.where(df.vents == "fine", -0.6, 0) + df.event.map(u)
    df["burnt"] = (rng.uniform(size=n) < expit(eta)).astype(int)
    return df


def test_unknown_carries_no_parameter():
    df = synth()
    df.loc[df.index[:500], "vents"] = T.UNKNOWN
    d = Design(cats=["vents"], use_num=False).fit(df)
    assert d.columns == ["vents=fine"]
    x = d.transform(df.iloc[:1])
    assert abs(x[0, 0] - d.freq["vents"]["fine"]) < 1e-12


def test_recovers_effect_and_tau():
    df = synth()
    m = VulnModel(Design(cats=["vents"], use_num=False)).fit(df)
    b = dict(zip(m.design.columns, m.beta))["vents=fine"]
    assert abs(b + 0.6) < 0.15, b
    assert 0.7 < np.sqrt(m.tau2) < 1.8, m.tau2


def test_live_update_moves_toward_data():
    df = synth()
    m = VulnModel(Design(cats=["vents"], use_num=False)).fit(df)
    obs = df.head(40).assign(burnt=1)
    u, v = m.event_posterior(obs)
    assert u > 0 and v < m.tau2


def test_commercial_interactions_only_touch_commercial():
    df = synth()
    df.loc[df.index[:800], "struct"] = "com_1"
    d = Design(cats=["struct", "vents"], use_num=False, interact=True).fit(df)
    assert d.columns[-1] == "com:vents=fine"
    X = d.transform(df)
    com = (df.struct == "com_1").to_numpy()
    assert np.all(X[~com, -1] == 0) and np.allclose(X[com, -1], X[com, d.columns.index("vents=fine")])


def test_credit_shrinks_with_severity():
    assert credit(-0.5, 0.3) > credit(-0.5, 0.9) > 0
    assert abs(credit(0.0, 0.4)) < 1e-12


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
