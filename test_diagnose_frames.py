"""python -m pytest -q test_diagnose_frames.py -- the rigid-fit diagnostics of diagnose_frames.py on constructed plans"""
import numpy as np
import pytest

import diagnose_frames as d


def cloud(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) * [3.0, 1.0] + [10, -4]


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_aligned_slices_fit_with_no_rotation_and_no_mismatch():
    X = cloud()
    P = np.eye(len(X)) / len(X)
    f = d.plan_rigid_fit(P, X, X + [500, 50])                       # a pure shift is still rigid
    assert abs(f["angle_deg"]) < 1e-6 and f["mismatch"] < 1e-6 and f["scale_ratio"] == pytest.approx(1.0)
    assert f["mismatch_mirror"] > 0.3
    assert d.partner_distance(P, X, X) == pytest.approx(0.0, abs=1e-9)
    assert d.partner_distance(P, X, X + [500, 50]) > 10             # raw distance to the partner is large


def test_a_rotated_target_is_recovered_with_its_angle():
    X = cloud()
    Y = (X - X.mean(0)) @ rot(90).T                                 # y = rot(90)(x - c), so mapping Y onto X is rot(-90)
    f = d.plan_rigid_fit(np.eye(len(X)) / len(X), X, Y)
    assert f["angle_deg"] == pytest.approx(-90.0, abs=1e-6) and f["mismatch"] < 1e-6


def test_a_mirrored_target_is_detected_and_scale_is_reported():
    X = cloud()
    Y = (X - X.mean(0)) * [1, -1]
    f = d.plan_rigid_fit(np.eye(len(X)) / len(X), X, Y)
    assert f["mismatch"] > 0.3 and f["mismatch_mirror"] < 1e-6
    big = d.plan_rigid_fit(np.eye(len(X)) / len(X), X, (X - X.mean(0)) * 2.0)
    assert big["scale_ratio"] == pytest.approx(2.0, rel=1e-6) and big["mismatch"] > 0.3     # rigid fits can't absorb scale


def test_random_plan_has_mismatch_near_one():
    rng = np.random.default_rng(1)
    X, Y = cloud(200, 1), cloud(220, 2)
    P = rng.random((200, 220))
    P /= P.sum()
    f = d.plan_rigid_fit(P, X, Y)
    assert 0.9 < f["mismatch"] <= 1.5 and 0.9 < f["mismatch_mirror"] <= 1.5


def test_unbalanced_plan_with_uneven_row_mass():
    X = cloud()
    P = np.diag(np.linspace(0.2, 2.0, len(X)))
    P /= P.sum()
    f = d.plan_rigid_fit(P, X, X + [1, 1])
    assert f["mismatch"] < 1e-6 and abs(f["angle_deg"]) < 1e-6


def test_main_prints_one_row_per_pair(tmp_path, monkeypatch, capsys):
    import anndata as ad
    import pandas as pd
    import call_moscot
    rng = np.random.default_rng(0)
    parts = []
    for t in (1.0, 2.0, 3.0):
        a = ad.AnnData(rng.normal(size=(40, 5)).astype(np.float32))
        a.obs["time"] = t
        a.obs["Annotation"] = pd.Categorical(["a"] * 40)
        a.obsm["spatial"] = rng.normal(size=(40, 2)) * 10 + t * 100
        a.obs_names = [f"{t}_{i}" for i in range(40)]
        parts.append(a)
    path = tmp_path / "d.h5ad"
    ad.concat(parts).write_h5ad(path)

    def fake_train(adata, time_key, spatial_key, *args, **kw):          # a plan pairing cell i with cell i
        ts = np.unique(adata.obs[time_key])
        src, tgt = adata[adata.obs[time_key] == ts[0]], adata[adata.obs[time_key] == ts[1]]
        return {"P": np.eye(40) / 40, "src": src, "tgt": tgt}

    monkeypatch.setattr(call_moscot, "train_moscot", fake_train)
    d.main(["--input", str(path), "--pairs", "1:2", "1:3"])
    out = capsys.readouterr().out
    assert "raw coordinates per slice" in out and "t=3: n=40" in out
    rows = [l for l in out.splitlines() if l.startswith(("1->2", "1->3"))]
    assert len(rows) == 2                                               # one summary row per pair
