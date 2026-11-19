"""
call_moscot.py -- train moscot on two timepoints and build an interpolated slice between them.

    model = train_moscot(support_adata, ...)          # support_adata has exactly 2 timepoints
    pred  = interpolate_moscot(model, target_time)    # AnnData of predicted cells at target_time

How the interpolation works (see the notebook / the maths write-up):
  1. moscot gives a coupling P between the two observed slices (fused GW, optionally unbalanced).
  2. Growth per source cell g_i comes from P's row sums; the number of cells at the target time is
     N = n_src^(1-lam) * n_tgt^lam  with  lam = (t - t_src) / (t_tgt - t_src).
  3. Draw N parents with probability proportional to g_i^lam; for each, draw a partner from that
     parent's row of P; place the new cell at (1-lam)*parent + lam*partner (space, expression, PCA).
  4. Cell types: moscot has no classifier, so a new cell takes the label of the nearer of its two source
     cells, i.e. the partner when lam > 0.5 (this includes forward extrapolation) and the parent otherwise.
     The labels of both source cells are kept as well (parent_label, partner_label).
"""
import numpy as np
import pandas as pd

from common import dense


def train_moscot(adata, time_key="time", spatial_key="spatial", n_pca=10,
                 epsilon=1e-2, alpha=0.5, tau=0.9, label_key="Annotation"):
    """
    Solve the moscot spatiotemporal problem between the two timepoints in `adata`.
    tau = 1.0 is balanced (no growth/death); below 1.0 lets moscot create/destroy mass.
    Returns a dict holding the coupling and the two slices in the coupling's row/column order.
    """
    import scanpy as sc
    from moscot.problems.spatiotemporal import SpatioTemporalProblem

    adata = adata.copy()                       # moscot converts the time column in place
    times = np.unique(np.asarray(adata.obs[time_key], dtype=float))
    if len(times) != 2:
        raise ValueError(f"moscot support data must have exactly 2 timepoints, got {times.tolist()}")

    sc.pp.pca(adata, n_comps=int(min(n_pca, adata.n_obs - 1, adata.n_vars - 1)))
    tp = SpatioTemporalProblem(adata)
    tp = tp.prepare(time_key=time_key, spatial_key=spatial_key, joint_attr="X_pca")
    tp = tp.solve(epsilon=epsilon, alpha=alpha, tau_a=tau, tau_b=tau)

    prob = tp[float(times[0]), float(times[1])]
    return {
        "P": np.asarray(prob.solution.transport_matrix),
        "src": prob.adata_src, "tgt": prob.adata_tgt,
        "t_src": float(times[0]), "t_tgt": float(times[1]),
        "spatial_key": spatial_key, "label_key": label_key,
        "params": dict(n_pca=n_pca, epsilon=epsilon, alpha=alpha, tau=tau),
    }


def interpolate_moscot(model, target_time, n_pred=None, jitter=0.0, seed=0, chunk=2000,
                       allow_extrapolation=False):
    """
    Build predicted cells at `target_time`.
    By default the target must lie strictly between the model's two timepoints (interpolation).
    With allow_extrapolation=True it may lie outside them:
      forward  (target > t_tgt): the same formulas with lam > 1, i.e. the straight line parent -> partner is
               continued past the partner, and the cell count keeps following n_src^(1-lam) * n_tgt^lam.
    Only forward extrapolation is supported: a target before t_src is always rejected.
    n_pred overrides the automatic cell count; jitter is the std of Gaussian noise added to positions.
    """
    import anndata as ad

    P, src, tgt = model["P"], model["src"], model["tgt"]
    t_src, t_tgt = model["t_src"], model["t_tgt"]
    spatial_key, label_key = model["spatial_key"], model["label_key"]
    if np.isclose(target_time, t_src) or np.isclose(target_time, t_tgt):
        raise ValueError(f"target {target_time} coincides with an observed timepoint ({t_src}, {t_tgt})")
    if target_time < t_src:
        raise ValueError(f"target {target_time} is before the first timepoint {t_src}; "
                         f"backward extrapolation is not supported")
    if target_time > t_tgt and not allow_extrapolation:
        raise ValueError(f"target {target_time} must lie between {t_src} and {t_tgt} "
                         f"(pass allow_extrapolation=True to go beyond {t_tgt})")
    rng = np.random.default_rng(seed)
    n_src, n_tgt = P.shape
    lam = (target_time - t_src) / (t_tgt - t_src)

    # growth per source cell (mean 1), and how many cells exist at the target time
    row_mass = P.sum(axis=1)
    g = n_src * row_mass / row_mass.sum()
    if n_pred is None:
        n_pred = int(round(n_src ** (1 - lam) * n_tgt ** lam))
    w = g ** lam
    parents = rng.choice(n_src, size=n_pred, p=w / w.sum())

    # one partner per descendant, drawn from its parent's row of P (chunked to bound memory)
    partners = np.empty(n_pred, dtype=int)
    for a in range(0, n_pred, chunk):
        b = min(a + chunk, n_pred)
        rows = P[parents[a:b]]
        rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-300)
        u = rng.random((b - a, 1))
        partners[a:b] = np.minimum((rows.cumsum(axis=1) < u).sum(axis=1), n_tgt - 1)

    def mix(a, b):
        return (1 - lam) * a + lam * b

    xy = mix(np.asarray(src.obsm[spatial_key])[parents], np.asarray(tgt.obsm[spatial_key])[partners])
    if jitter > 0:
        xy = xy + rng.normal(scale=jitter, size=xy.shape)
    X = mix(dense(src.X)[parents], dense(tgt.X)[partners])
    pcs = mix(np.asarray(src.obsm["X_pca"])[parents], np.asarray(tgt.obsm["X_pca"])[partners])

    parent_label = src.obs[label_key].astype(str).values[parents]
    partner_label = tgt.obs[label_key].astype(str).values[partners]
    # the new cell sits lam of the way from parent to partner, so it inherits the nearer one's label
    # (lam > 0.5 also covers forward extrapolation, where the partner is the nearest observed cell)
    labels = partner_label if lam > 0.5 else parent_label

    pred = ad.AnnData(X=X.astype(np.float32))
    pred.var_names = list(src.var_names)
    pred.obs_names = [f"moscot_t{target_time:g}_{i}" for i in range(n_pred)]
    pred.obsm["spatial"] = xy.astype(np.float32)          # raw coordinate units, same frame as the input
    pred.obsm["X_pca"] = pcs.astype(np.float32)
    pred.obs[label_key] = pd.Categorical(labels)
    pred.obs["parent_label"] = parent_label
    pred.obs["partner_label"] = partner_label
    pred.obs["parent_idx"] = parents
    pred.obs["partner_idx"] = partners
    return pred