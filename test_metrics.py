"""
test_metrics.py -- hand-checkable sanity tests for interp_metrics.py on tiny point clouds.

    python -m pytest -q test_metrics.py

Each test builds a few points where the right answer is known (identical clouds -> 0, a pure shift -> its
length, a rotation -> only the aligned/GW metrics stay 0, ...).
"""
import warnings

import anndata as ad
import numpy as np
import ot
import pytest

from interp_metrics import compute_metrics, evaluate_interpolation

W, ALPHA = 0.9, 0.5
# 6 points with no mirror symmetry and no rotational symmetry (so rotations/reflections are detectable)
S0 = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [0, 4]], dtype=float)
X0 = np.array([[0, 0, 1], [1, 0, 0], [0, 2, 0], [3, 1, 0], [0, 1, 4], [2, 2, 2]], dtype=float)


def rot(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def metrics(Xp=X0, Sp=S0, Xt=X0, St=S0, **kw):
    return compute_metrics(Xp, Sp, Xt, St, alpha=ALPHA, w=W, **kw)


# ------------------------------------------------------------------ identical / reordered clouds
def test_identical_is_zero():
    m = metrics()
    for k in ("w2_expr", "w2_joint", "w2_joint_aligned", "gw_spatial", "fgw"):
        assert m[k] == pytest.approx(0, abs=1e-5), k
    assert m["frit_angle_deg"] == pytest.approx(0, abs=1e-3)


def test_row_order_does_not_matter():
    perm = np.random.default_rng(1).permutation(len(X0))
    m = metrics(Xp=X0[perm], Sp=S0[perm])
    for k in ("w2_expr", "w2_joint", "w2_joint_aligned", "gw_spatial", "fgw"):
        assert m[k] == pytest.approx(0, abs=1e-5), k


def test_duplicated_cells_with_uniform_weights_is_zero():
    m = metrics(Xp=np.vstack([X0, X0]), Sp=np.vstack([S0, S0]))      # 12 predicted vs 6 true, each twice
    assert m["w2_expr"] == pytest.approx(0, abs=1e-6)
    assert m["w2_joint"] == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------ expression only
def test_w2_expr_known_1d_value():
    # {0,3} vs {0,1}: best matching 0-0, 3-1 -> W2^2 = (0 + 4)/2 = 2
    Xp, Xt = np.array([[0.], [3.]]), np.array([[0.], [1.]])
    S = np.array([[0., 0.], [1., 0.]])
    assert metrics(Xp, S, Xt, S)["w2_expr"] == pytest.approx(np.sqrt(2))


def test_expression_shift_equals_its_length():
    d = np.array([1.0, 2.0, 2.0])                                     # |d| = 3
    m = metrics(Xp=X0 + d)
    assert m["w2_expr"] == pytest.approx(3.0, abs=1e-6)
    assert m["w2_joint"] == pytest.approx(np.sqrt((1 - W) * 9.0), abs=1e-6)      # space term is 0 for the identity plan
    # spatial structure identical -> gw is 0; fgw is only the feature term: (1-alpha) * |d|^2 / m_scale
    m_scale = ot.dist(X0, X0).mean()
    assert m["gw_spatial"] == pytest.approx(0, abs=1e-6)
    assert m["fgw"] == pytest.approx((1 - ALPHA) * 9.0 / m_scale, rel=1e-4)
    assert m["w2_joint_aligned"] == pytest.approx(np.sqrt((1 - W) * 9.0), abs=1e-4)


def test_larger_expression_shift_scores_worse():
    a, b = metrics(Xp=X0 + 1.0), metrics(Xp=X0 + 2.0)
    for k in ("w2_expr", "w2_joint", "fgw"):
        assert b[k] > a[k], k


def test_expression_metrics_ignore_space_and_vice_versa():
    base = metrics(Xp=X0 + 1.0)
    moved = metrics(Xp=X0 + 1.0, Sp=S0 + [50, -20])                   # change only the predicted coordinates
    assert moved["w2_expr"] == pytest.approx(base["w2_expr"], abs=1e-6)
    shuffled_expr = metrics(Xp=X0[::-1].copy() * 1.0, Sp=S0 + 0.0)    # gw only depends on space
    assert shuffled_expr["gw_spatial"] == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------ space: translation / rotation / reflection / scale
def test_spatial_translation():
    m = metrics(Sp=S0 + [3, 4])                                       # shift by length 5
    assert m["w2_expr"] == pytest.approx(0, abs=1e-6)
    assert m["w2_joint"] == pytest.approx(np.sqrt(W * 25.0), abs=1e-5)   # unaligned metric feels the shift
    assert m["w2_joint_aligned"] == pytest.approx(0, abs=1e-4)           # aligned ignores translation
    assert m["gw_spatial"] == pytest.approx(0, abs=1e-6)
    assert m["fgw"] == pytest.approx(0, abs=1e-6)
    assert m["frit_angle_deg"] == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize("deg", [30, 90, 180])
def test_spatial_rotation(deg):
    m = metrics(Sp=S0 @ rot(deg).T)
    assert m["w2_joint"] > 0.5                                        # unaligned metric is hurt
    assert m["w2_joint_aligned"] == pytest.approx(0, abs=1e-3)        # rigid alignment undoes it
    assert m["gw_spatial"] == pytest.approx(0, abs=1e-6)
    assert m["fgw"] == pytest.approx(0, abs=1e-6)
    assert abs(m["frit_angle_deg"]) == pytest.approx(deg, abs=0.5)    # recovered angle


def test_spatial_reflection():
    m = metrics(Sp=S0 * np.array([-1, 1]))
    assert m["gw_spatial"] == pytest.approx(0, abs=1e-6)              # GW is reflection invariant
    assert m["fgw"] == pytest.approx(0, abs=1e-6)
    assert m["w2_joint_aligned"] > 1e-2                               # FRIT forbids reflections
    assert m["w2_joint"] > 1e-2


def test_spatial_scale_is_detected():
    m = metrics(Sp=S0 * 2.0)
    assert m["w2_expr"] == pytest.approx(0, abs=1e-6)
    assert m["gw_spatial"] > 1e-3                                     # GW is NOT scale invariant
    assert m["w2_joint_aligned"] > 1e-2
    assert m["w2_joint"] > 1e-2


def test_gw_normalised_by_true_slice_only():
    # multiplying BOTH slices' coordinates by 10 must not change gw (costs divided by the truth's mean distance)
    a = metrics(Sp=S0 * 1.5, St=S0)
    b = metrics(Sp=S0 * 15.0, St=S0 * 10.0)
    assert b["gw_spatial"] == pytest.approx(a["gw_spatial"], rel=1e-4)


def test_w2_symmetry():
    Sp = S0 + [1, 2]
    a = compute_metrics(X0 + 1.0, Sp, X0, S0, alpha=ALPHA, w=W, gw=False, frit=False)
    b = compute_metrics(X0, S0, X0 + 1.0, Sp, alpha=ALPHA, w=W, gw=False, frit=False)
    assert a["w2_expr"] == pytest.approx(b["w2_expr"])
    assert a["w2_joint"] == pytest.approx(b["w2_joint"])


def test_switches_skip_metrics():
    m = metrics(gw=False, frit=False)
    assert set(m) == {"w2_expr", "w2_joint"}


# ------------------------------------------------------------------ end to end: evaluate_interpolation on AnnData
def make_original():
    rng = np.random.default_rng(0)
    blocks = []
    for t in (1.0, 2.0, 3.0):
        a = ad.AnnData(X=(X0 + (t - 2) * 1.5 + rng.normal(scale=0.01, size=X0.shape)).astype(np.float32))
        a.var_names = ["g1", "g2", "g3"]
        a.obs_names = [f"t{t:g}_{i}" for i in range(len(X0))]
        a.obsm["spatial"] = (S0 + (t - 2) * np.array([3.0, 0.0]) + rng.normal(scale=0.01, size=S0.shape))
        a.obs["time"] = t
        blocks.append(a)
    return ad.concat(blocks)


def run(pred, original, **kw):
    base = dict(time=2.0, n_max=100, n_repeats=2, baseline_times=(1.0, 3.0))
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                # a unit-mismatch warning would be a bug here
        return evaluate_interpolation(pred, original, **base)


def test_evaluate_perfect_prediction_beats_baselines():
    orig = make_original()
    pred = orig[orig.obs["time"] == 2.0].copy()
    pred.obs_names = [f"p{i}" for i in range(pred.n_obs)]
    res = run(pred, orig)
    assert res["n_pred"] == res["n_true"] == 6
    for k in ("w2_expr", "w2_joint", "w2_joint_aligned", "gw_spatial", "fgw"):
        assert res["pred"][k][0] == pytest.approx(0, abs=1e-3), k
    for bt in (1.0, 3.0):                                             # un-interpolated neighbours are clearly worse
        for k in ("w2_expr", "w2_joint"):
            assert res["baselines"][bt][k][0] > 1.0, (bt, k)


def test_evaluate_center_hides_translation_only_when_asked():
    orig = make_original()
    pred = orig[orig.obs["time"] == 2.0].copy()
    pred.obsm["spatial"] = pred.obsm["spatial"] + np.array([5.0, 5.0])
    off = run(pred, orig, center=False)["pred"]["w2_joint"][0]
    on = run(pred, orig, center=True)["pred"]["w2_joint"][0]
    assert off == pytest.approx(np.sqrt(W * 50.0), abs=0.05)
    assert on == pytest.approx(0, abs=1e-3)


def test_evaluate_uses_shared_genes_only():
    orig = make_original()
    pred = orig[orig.obs["time"] == 2.0].copy()
    pred = pred[:, ["g3", "g1"]].copy()                               # reordered and missing g2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evaluate_interpolation(pred, orig, 2.0, n_max=100, n_repeats=1)
    assert res["pred"]["w2_expr"][0] == pytest.approx(0, abs=1e-3)    # gene order must be matched by name


def test_evaluate_no_shared_genes_raises():
    orig = make_original()
    pred = orig[orig.obs["time"] == 2.0].copy()
    pred.var_names = ["x1", "x2", "x3"]
    with pytest.raises(ValueError):
        evaluate_interpolation(pred, orig, 2.0, n_max=100, n_repeats=1)
