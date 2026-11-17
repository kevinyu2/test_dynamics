"""
interp_metrics.py -- score a predicted timepoint against the held-out truth.

Four numbers, all comparing the PREDICTED cells with the TRUE cells at the
held-out time (each slice treated as a uniform distribution over cells).

The first two are the leave-one-time-point-out metrics of the stVCR paper
(Peng, Zhou & Li, Nat. Methods 2026, Methods: "Quantitative metric used in
benchmark experiments"):

  w2_expr    Wasserstein-2 distance between the expression clouds
             (cost = squared Euclidean distance between expression vectors).
  w2_joint   Wasserstein-2 with the joint cost
                 c_ij = (1 - w) * ||q_i - q_j||^2 + w * ||x_i - x_j||^2,   w = 0.9 in the paper.
             This is plain (linear) OT, NOT Gromov-Wasserstein, so it needs both
             slices in ONE aligned coordinate frame, and it depends on the units
             of x and q (the paper doesn't say which; see truth_spatial_scale).

  w2_joint_aligned
             The same joint cost, but with the TRUE slice rigidly moved (rotation +
             translation, no reflection) to best match the prediction -- the paper's
             "fused rigid-body-transformation-invariant OT" (FRIT-OT, Supplementary
             Note 1, Eq. 10), which Supplementary Tables 5/9/11 use for real data.
             It is ONE OT problem with one coupling: an alternating minimisation of
             (1) the OT plan for a fixed transform and (2) the transform for a fixed
             plan (weighted Procrustes). frit_angle_deg reports the fitted 2D rotation
             as a diagnostic: a large angle means the frames did not agree.

The last two are NOT in the paper (extras):
  gw_spatial Gromov-Wasserstein between the two spatial layouts
             (compares distances WITHIN each slice, so no alignment needed).
  fgw        Fused GW: expression is the cross-slice feature cost, spatial
             distances are the within-slice structure.
             objective = (1-alpha) * <M, T> + alpha * sum (C1_ik - C2_jl)^2 T_ij T_kl

For gw_spatial and fgw all costs are divided by constants taken from the TRUE
slice only (its mean squared expression distance / its mean spatial distance),
so values are comparable between different methods' predictions.

Invariances: gw_spatial and fgw ignore translation, rotation AND reflection of
the spatial coordinates, but NOT scale. w2_joint_aligned ignores translation and
rotation (but not reflection or scale). w2_joint ignores nothing: the truth must
be in the same frame AND units as the prediction (see truth_transform, center,
truth_spatial_scale below).

Also reports the same metrics between the truth and the observed neighbouring
timepoints ("baselines"), which tells you what "no interpolation" would score.

Needs: numpy, scipy, anndata, pot  (pip install pot)

Library use:
    from interp_metrics import evaluate_interpolation, format_report
    res = evaluate_interpolation(pred, original, time=3.0)
    print(format_report(res))

Command line:
    python interp_metrics.py --pred moscot_pred_t3.h5ad --original toy_all.h5ad \
        --time 3.0 --baseline-times 2 4
"""
import argparse
import json
import warnings

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------- core math
def _procrustes(X, Y, P):
    """Proper rotation R and shift r minimising sum_ij P_ij ||x_i - (R y_j + r)||^2."""
    p, q, tot = P.sum(1), P.sum(0), P.sum()
    xb, yb = p @ X / tot, q @ Y / tot
    U, _, Vt = np.linalg.svd((X - xb).T @ P @ (Y - yb))
    D = np.eye(len(U))
    D[-1, -1] = np.sign(np.linalg.det(U @ Vt))      # forbid reflections
    R = U @ D @ Vt
    return R, xb - R @ yb


def frit_ot(Mexpr, Sp, St, p, q, w=0.9, max_iter=30, tol=1e-6):
    """
    Fused rigid-transformation-invariant OT (stVCR Supp. Note 1, Eq. 10), with the TRUE
    slice (St) as the set that is moved:
        min over plan P and rigid T of  sum_ij P_ij * [ w ||x_i - T(y_j)||^2 + (1-w) Mexpr_ij ].
    Alternates the OT plan (T fixed) and the Procrustes fit of T (plan fixed); the objective
    never increases, but T can land in a local minimum, so in 2D we start from 4 rotations
    (0/90/180/270 degrees) with centroids matched and keep the best. Returns
    (objective, R, r).
    """
    import ot
    d = Sp.shape[1]
    if d == 2:
        starts = [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                  for a in np.deg2rad([0, 90, 180, 270])]
    else:
        starts = [np.eye(d)]
    best = None
    for R in starts:
        r = Sp.mean(0) - R @ St.mean(0)
        prev = np.inf
        for _ in range(max_iter):
            Cspa = ot.dist(Sp, St @ R.T + r)
            P = ot.emd(p, q, np.ascontiguousarray(w * Cspa + (1 - w) * Mexpr), numItermax=1_000_000)
            obj = float((P * (w * Cspa + (1 - w) * Mexpr)).sum())
            if prev - obj < tol * max(abs(obj), 1e-12):
                break
            prev = obj
            R, r = _procrustes(Sp, St, P)
        if best is None or obj < best[0]:
            best = (obj, R, r)
    return best


def compute_metrics(Xp, Sp, Xt, St, alpha=0.5, w=0.9, gw=True, frit=True):
    """Metrics between predicted (Xp expression, Sp space) and true (Xt, St) clouds."""
    import ot

    n, m = len(Xp), len(Xt)
    p = np.full(n, 1.0 / n)
    q = np.full(m, 1.0 / m)

    M = ot.dist(Xp, Xt)                              # (n, m) squared Euclidean
    if gw:
        Cp = ot.dist(Sp, Sp, metric="euclidean")     # (n, n) within-slice distances
        Ct = ot.dist(St, St, metric="euclidean")     # (m, m)
        m_scale = ot.dist(Xt, Xt).mean()             # scales from the TRUTH only
        c_scale = Ct.mean()

    out = {"w2_expr": float(np.sqrt(ot.emd2(p, q, M)))}
    # paper's joint metric: linear OT on (1-w)*expr + w*space squared distances (raw units)
    out["w2_joint"] = float(np.sqrt(ot.emd2(p, q, (1 - w) * M + w * ot.dist(Sp, St))))
    if frit:
        obj, R, _ = frit_ot(M, Sp, St, p, q, w)
        out["w2_joint_aligned"] = float(np.sqrt(obj))
        if R.shape == (2, 2):
            out["frit_angle_deg"] = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    if gw:
        out["gw_spatial"] = float(ot.gromov.gromov_wasserstein2(Cp / c_scale, Ct / c_scale, p, q, "square_loss"))
        out["fgw"] = float(ot.gromov.fused_gromov_wasserstein2(
            M / m_scale, Cp / c_scale, Ct / c_scale, p, q, "square_loss", alpha=alpha))
    return out


# ------------------------------------------------------------ data handling
def _dense(x):
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _expr(a, expr_key, genes):
    if expr_key is None:
        return _dense(a[:, genes].X).astype(np.float64)
    return np.asarray(a.obsm[expr_key], dtype=np.float64)


def _slice_at(original, time_key, time):
    t = np.asarray(original.obs[time_key], dtype=float)
    mask = np.isclose(t, float(time))
    if not mask.any():
        raise ValueError(f"No cells with {time_key} == {time}. Available: {np.unique(t)}")
    return original[mask]


def _subsample(n, n_max, rng):
    return np.arange(n) if n <= n_max else np.sort(rng.choice(n, n_max, replace=False))


def _spread(a):
    return float(np.mean(np.std(a, axis=0)))


def evaluate_interpolation(pred, original, time, time_key="time", spatial_key="spatial",
                           pred_spatial_key=None, expr_key=None, truth_spatial_scale=1.0,
                           alpha=0.5, n_max=2000, n_repeats=3, seed=0, baseline_times=(),
                           w=0.9, center=False, truth_transform=None, gw=True, frit=True, genes=None):
    """
    pred                 AnnData with the predicted cells (X or obsm[expr_key], obsm[pred_spatial_key]).
    original             AnnData with ALL real timepoints (must contain `time`).
    time                 the held-out timepoint, in the units of original.obs[time_key].
    expr_key             None -> compare .X on the genes both objects share (var_names must match!).
                         Otherwise an obsm key present in both, in the SAME basis (e.g. a shared PCA).
    pred_spatial_key     obsm key for predicted coordinates (default: spatial_key).
    w                    spatial weight in the joint cost (paper: 0.9).
    center               subtract each slice's own centroid (pred, truth, baselines), as stVCR's
                         preprocessing does. Use it to compare a raw-frame prediction (e.g. moscot)
                         with a stVCR-style one. Hides a wrong overall position of the slice.
    truth_transform      optional (R, r): true/baseline coordinates become x @ R.T + r BEFORE
                         scaling, e.g. a rigid alignment into the prediction's frame. How the paper
                         aligns the held-out slice is in its Supplementary Note 1; don't fit the
                         transform to the prediction itself, that favours the prediction.
    gw                   False skips gw_spatial/fgw (the expensive, non-paper metrics).
    frit                 False skips w2_joint_aligned (an alternating OT + Procrustes loop; also costly).
    genes                explicit list of gene names to compare (default: the genes pred and original share).
                         Use the same list for every model so their expression scores are comparable.
    truth_spatial_scale  true (and baseline) coordinates are divided by this before comparing.
                         Use 1.0 for moscot-style raw-frame predictions;
                         for stVCR use adata.uns['spatial_scale_factor'].
    n_max, n_repeats     GW is O(n^3): each repeat subsamples up to n_max cells per slice.
    baseline_times       observed timepoints to score against the truth for reference.
    Returns a dict with mean/std over repeats.
    """
    truth = _slice_at(original, time_key, time)
    pred_spatial_key = pred_spatial_key or spatial_key

    if expr_key is None:
        if genes is None:
            shared = set(original.var_names)
            genes = [g for g in pred.var_names if g in shared]
        else:
            genes = list(genes)
        if not genes:
            raise ValueError("pred and original share no gene names. If pred came from stVCR, set "
                             "pred.var_names to the gene names of the (HVG-filtered) training adata.")
        if len(genes) < pred.n_vars and set(genes) != set(pred.var_names) & set(genes):
            warnings.warn(f"only {len(genes)}/{pred.n_vars} predicted genes found in original")

    Xp, Sp = _expr(pred, expr_key, genes), np.asarray(pred.obsm[pred_spatial_key], dtype=np.float64)
    Xt = _expr(truth, expr_key, genes)
    def _truth_xy(a):
        xy = np.asarray(a.obsm[spatial_key], dtype=np.float64)
        if truth_transform is not None:
            R, r = truth_transform
            xy = xy @ np.asarray(R, dtype=np.float64).T + np.asarray(r, dtype=np.float64)
        return xy / truth_spatial_scale

    St = _truth_xy(truth)
    if center:
        Sp, St = Sp - Sp.mean(0), St - St.mean(0)

    # sanity checks: mismatched units are the most common way to get a meaningless score
    for name, a, b in (("spatial", Sp, St), ("expression", Xp, Xt)):
        r = _spread(a) / max(_spread(b), 1e-12)
        if not 0.5 < r < 2.0:
            warnings.warn(f"{name} spread of pred is {r:.2f}x that of the truth -- "
                          f"are the two in the same units?")

    refs = {}
    for bt in baseline_times:
        s = _slice_at(original, time_key, bt)
        Sb = _truth_xy(s)
        refs[bt] = (_expr(s, expr_key, genes), Sb - Sb.mean(0) if center else Sb)

    rng = np.random.default_rng(seed)
    runs, ref_runs = [], {bt: [] for bt in refs}
    for _ in range(n_repeats):
        ip, it = _subsample(len(Xp), n_max, rng), _subsample(len(Xt), n_max, rng)
        runs.append(compute_metrics(Xp[ip], Sp[ip], Xt[it], St[it], alpha, w, gw, frit))
        for bt, (Xb, Sb) in refs.items():
            ib = _subsample(len(Xb), n_max, rng)
            ref_runs[bt].append(compute_metrics(Xb[ib], Sb[ib], Xt[it], St[it], alpha, w, gw, frit))

    def agg(rs):
        return {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}

    return {"time": float(time), "alpha": alpha, "w": w, "n_pred": int(len(Xp)), "n_true": int(len(Xt)),
            "n_max": n_max, "n_repeats": n_repeats,
            "pred": agg(runs), "baselines": {bt: agg(r) for bt, r in ref_runs.items()}}


def format_report(res):
    keys = [k for k in ("w2_expr", "w2_joint", "w2_joint_aligned", "frit_angle_deg", "gw_spatial", "fgw") if k in res["pred"]]
    lines = [f"held-out t={res['time']:g} | n_pred={res['n_pred']} n_true={res['n_true']} | "
             f"w={res['w']} alpha={res['alpha']} | {res['n_repeats']} repeats of <= {res['n_max']} cells",
             f"{'':>18}" + "".join(f"{k:>18}" for k in keys)]
    def row(name, d):
        return f"{name:>18}" + "".join(f"{d[k][0]:>11.4f}+-{d[k][1]:<5.4f}" for k in keys)
    lines.append(row("prediction", res["pred"]))
    for bt, d in res["baselines"].items():
        lines.append(row(f"observed t={bt:g}", d))
    return "\n".join(lines)


# ------------------------------------------------------------------------ CLI
def main():
    import anndata as ad
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predicted cells (.h5ad)")
    ap.add_argument("--original", required=True, help="original data with all timepoints (.h5ad)")
    ap.add_argument("--time", type=float, required=True, help="held-out timepoint")
    ap.add_argument("--time-key", default="time")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--pred-spatial-key", default=None)
    ap.add_argument("--expr-key", default=None, help="obsm key; default compares .X on shared genes")
    ap.add_argument("--truth-spatial-scale", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="FGW structure weight (extra metric)")
    ap.add_argument("--w", type=float, default=0.9, help="spatial weight in joint W2 (paper: 0.9)")
    ap.add_argument("--center", action="store_true", help="centre each slice's coordinates")
    ap.add_argument("--no-gw", action="store_true", help="skip gw_spatial / fgw")
    ap.add_argument("--no-frit", action="store_true", help="skip w2_joint_aligned (rigid-aligned joint W2)")
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-times", type=float, nargs="*", default=[])
    ap.add_argument("--out", default=None, help="write results as JSON")
    a = ap.parse_args()

    res = evaluate_interpolation(
        ad.read_h5ad(a.pred), ad.read_h5ad(a.original), a.time, a.time_key, a.spatial_key,
        a.pred_spatial_key, a.expr_key, a.truth_spatial_scale, a.alpha, a.n_max, a.n_repeats,
        a.seed, a.baseline_times, a.w, a.center, None, not a.no_gw, not a.no_frit)
    print(format_report(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()