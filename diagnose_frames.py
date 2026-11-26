#!/usr/bin/env python
"""
diagnose_frames.py -- are two slices in the same coordinate frame, and does moscot's plan respect it?

moscot builds an interpolated cell as (1-lambda)*parent + lambda*partner in RAW coordinates, but its plan (fused GW)
does not care how the two slices are placed, rotated or scaled relative to each other. If the plan pairs a cell with a
partner that is not near it in raw coordinates, the average lands somewhere in between (a squeezed, mixed-up slice).
For each pair  a:b  this script
  * prints each slice's centre and spread in raw coordinates,
  * solves the same moscot plan as the pipeline and reports how far partners are from their parents,
  * fits the best rigid transform (rotation + shift, optionally mirrored) that maps the target slice onto the source
    slice under the plan, and reports its angle and how well it fits.
A pair that is fine has a small rotation, a low mismatch and no mirror; a pair that breaks the interpolation does not.

    python diagnose_frames.py --input artista_2_20dpi.h5ad --pairs 2:20 2:15 5:20 5:15 \
        --moscot-epsilon 0.03 --moscot-tau 0.97 --n-cells 3000
Run it with the moscot environment's python. Without --n-cells it uses every cell, like the pipeline.
"""
import argparse

import numpy as np


def _wcov(P, X):
    p = P.sum(1)
    c = p @ X / p.sum()
    return c, float((p * ((X - c) ** 2).sum(1)).sum() / p.sum())


def plan_rigid_fit(P, X, Y):
    """
    Best proper rotation R and shift r mapping the TARGET coordinates Y onto the SOURCE coordinates X under the plan P
    (minimises sum_ij P_ij |x_i - (R y_j + r)|^2), also with Y mirrored. Returns a dict:
      angle_deg        rotation of the best proper fit
      mismatch         sqrt(weighted residual) / weighted spread of X: 0 = partners sit exactly where a rigid move
                       of the target slice puts them, ~1 = no better than ignoring the partner
      mismatch_mirror  the same with Y mirrored (y -> -y); much smaller than `mismatch` means the plan flips the slice
      scale_ratio      weighted spread of Y over that of X (the plan is blind to this)
    """
    from interp_metrics import _procrustes
    P = np.asarray(P, dtype=np.float64)
    X, Y = np.asarray(X, dtype=np.float64)[:, :2], np.asarray(Y, dtype=np.float64)[:, :2]
    tot = P.sum()
    cx, var_x = _wcov(P, X)
    cy, var_y = _wcov(P.T, Y)
    X, Y = X - cx, Y - cy              # a shift is absorbed by r; centring avoids cancellation in the residual below

    def fit(Yc):
        R, r = _procrustes(X, Yc, P)
        # residual sum_ij P_ij |x_i - R y_j - r|^2 = sum_i p_i|x_i|^2 + sum_j q_j|y_j|^2 - 2 sum_ij P_ij x_i.(R y_j + r) ...
        # computed through the barycentric partner to avoid forming the n x m matrix
        B = (P @ Yc) @ R.T + r * P.sum(1, keepdims=True)                # sum_j P_ij (R y_j + r)
        p = P.sum(1)
        q = P.sum(0)
        res = (p * (X ** 2).sum(1)).sum() + (q * (Yc ** 2).sum(1)).sum() + tot * float(r @ r) \
            + 2 * float(q @ Yc @ R.T @ r) - 2 * float((X * B).sum())
        return R, np.sqrt(max(res, 0.0) / tot / max(var_x, 1e-12))

    R, m = fit(Y)
    _, m_mirror = fit(Y * np.array([1.0, -1.0]))
    return dict(angle_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), mismatch=float(m),
                mismatch_mirror=float(m_mirror), scale_ratio=float(np.sqrt(var_y / max(var_x, 1e-12))))


def partner_distance(P, X, Y):
    """RMS distance between each source cell and its plan-averaged partner, relative to the slice's spread."""
    p = P.sum(1)
    B = (P @ Y[:, :2]) / np.maximum(p[:, None], 1e-30)
    _, var_x = _wcov(P, X[:, :2])
    return float(np.sqrt((p * ((B - X[:, :2]) ** 2).sum(1)).sum() / p.sum() / max(var_x, 1e-12)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="source:target timepoints, e.g. 2:20 5:15")
    p.add_argument("--time-key", default="time")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--label-key", default="Annotation")
    p.add_argument("--n-cells", type=int, default=None, help="subsample this many cells per timepoint")
    p.add_argument("--moscot-epsilon", type=float, default=1e-2)
    p.add_argument("--moscot-tau", type=float, default=0.9)
    p.add_argument("--moscot-alpha", type=float, default=0.5)
    p.add_argument("--n-pca", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    import anndata as ad
    from call_moscot import train_moscot
    import common

    full = ad.read_h5ad(a.input)
    pairs = [tuple(float(x) for x in s.split(":")) for s in a.pairs]
    times = sorted({t for pr in pairs for t in pr})
    adata, _ = common.select_times(full, a.time_key, times, a.n_cells, a.seed)
    t = np.asarray(adata.obs[a.time_key], dtype=float)
    print("raw coordinates per slice")
    for tt in times:
        xy = np.asarray(adata.obsm[a.spatial_key])[np.isclose(t, tt)][:, :2].astype(float)
        print(f"  t={tt:g}: n={len(xy)}  centre=({xy[:, 0].mean():.0f}, {xy[:, 1].mean():.0f})  "
              f"std=({xy[:, 0].std():.0f}, {xy[:, 1].std():.0f})  x {xy[:, 0].min():.0f}..{xy[:, 0].max():.0f}  "
              f"y {xy[:, 1].min():.0f}..{xy[:, 1].max():.0f}")
    rows = []
    for s, e in pairs:
        sub = adata[np.isin(t, [s, e])].copy()
        print(f"\nplan {s:g} -> {e:g}")
        m = train_moscot(sub, a.time_key, a.spatial_key, a.n_pca, a.moscot_epsilon, a.moscot_alpha, a.moscot_tau,
                         a.label_key)
        P, X, Y = m["P"], np.asarray(m["src"].obsm[a.spatial_key]), np.asarray(m["tgt"].obsm[a.spatial_key])
        g = P.sum(1) / P.sum(1).mean()
        fit = plan_rigid_fit(P, X, Y)
        rows.append((f"{s:g}->{e:g}", partner_distance(P, X, Y), fit, float(g.min()), float(g.max())))
    print("\npair      partner distance*  rigid angle  mismatch  mismatch if mirrored  spread ratio**  g min..max")
    for name, d, f, gmin, gmax in rows:
        print(f"{name:9s} {d:17.2f}  {f['angle_deg']:10.1f}  {f['mismatch']:8.2f}  {f['mismatch_mirror']:20.2f}  "
              f"{f['scale_ratio']:13.2f}  {gmin:.2f}..{gmax:.2f}")
    print("* RMS distance parent -> plan-averaged partner / spread of the slice (0 = partners sit on their parents)\n"
          "** spread of the target slice / spread of the source slice, in raw units")


if __name__ == "__main__":
    main()
