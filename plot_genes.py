#!/usr/bin/env python
"""
plot_genes.py -- look at one gene's expression in the predicted slice next to the real ones.

For every method (moscot / stvcr_two / stvcr_all) that has a prediction for the target, and every gene, it writes
one PNG with the cells coloured by that gene's expression, all panels on ONE colour scale:

  interpolation (two neighbours):  [ left slice | TRUE target | PREDICTED target | right slice ]
  extrapolation (one neighbour):   [ left slice | TRUE target | PREDICTED target ]

    python plot_genes.py --original artista_2_20dpi.h5ad --pred-dir axolotl_full/predictions \
        --target 10 --neighbors 5 15 --genes Sox2 Gfap --out axolotl_full/gene_plots
    python plot_genes.py --original artista_2_20dpi.h5ad --pred-dir axolotl_full/predictions \
        --target 20 --neighbors 15 --genes Sox2                      # one neighbour = forward extrapolation

Two neighbours mean "interpolation": the prediction file read is <model>__t<target>.h5ad. One neighbour means
"forward extrapolation": <model>__fwd_t<target>.h5ad. The neighbours only decide which real slices are drawn.

Notes
  * The colour range comes from the REAL panels only (0.5th to 99.5th percentile of the pooled values), so a prediction
    that is too high or too low saturates instead of stretching the scale. Points are drawn low to high so the
    expressing cells are on top. Each title shows the mean expression, which does not depend on coordinates.
  * Each panel is centred on its own centroid (as the cell-type images are). stVCR works in an aligned frame, so its
    panel can differ from the real slice by a rotation; --no-center draws raw coordinates instead.
  * stVCR only predicts its highly variable genes; a gene that a model did not predict is reported and skipped for it.
"""
import argparse
import os
import sys

import numpy as np

import common


def resolve_gene(name, var_names):
    """Exact match, else case-insensitive. Returns the name as spelled in var_names or None."""
    names = list(var_names)
    if name in set(names):
        return name
    low = {n.lower(): n for n in names}
    return low.get(name.lower())


def suggest(name, var_names, k=8):
    return [n for n in var_names if name.lower() in n.lower()][:k]


def gene_values(adata, gene):
    return np.asarray(common.dense(adata[:, gene].X), dtype=float).ravel()


def load_slices(path, time_key, spatial_key, times):
    """Real slices at `times`, read into memory one at a time (the file can be large)."""
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    t = np.asarray(a.obs[time_key], dtype=float)
    if spatial_key not in a.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found in {path}; found {list(a.obsm.keys())}")
    out = {}
    for tt in times:
        mask = np.isclose(t, float(tt))
        if not mask.any():
            raise ValueError(f"no cells with {time_key} == {tt:g} in {path}; available: {np.unique(t).tolist()}")
        out[tt] = a[mask].to_memory()
    return out


def check_times(target, neighbors):
    nb = sorted(float(x) for x in neighbors)
    if len(nb) == 2:
        if not nb[0] < target < nb[1]:
            raise ValueError(f"interpolation needs left < target < right, got {nb[0]:g}, {target:g}, {nb[1]:g}")
    elif len(nb) == 1:
        if not nb[0] < target:
            raise ValueError(f"forward extrapolation needs the neighbour to be earlier than the target "
                             f"({nb[0]:g} is not before {target:g}); backward extrapolation is not supported")
    else:
        raise ValueError("give one neighbour (extrapolation) or two (interpolation)")
    return nb


def color_range(values_list):
    pool = np.concatenate([np.asarray(v, dtype=float) for v in values_list])
    lo, hi = np.percentile(pool, 0.5), np.percentile(pool, 99.5)
    if not hi > lo:
        hi = lo + max(float(np.abs(pool).max()) * 1e-3, 1e-6)
    return float(lo), float(hi)


def build_figure(panels, gene, vlim, point_size=3.0, max_points=40000, center=True, cmap="viridis", seed=0, title=""):
    """panels: list of dicts(title, xy, vals). Returns the matplotlib figure (not saved)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    prepared = []
    for p in panels:
        xy = np.asarray(p["xy"], dtype=float)[:, :2]
        v = np.asarray(p["vals"], dtype=float)
        if center:
            xy = xy - xy.mean(0)
        if len(xy) > max_points:
            keep = rng.choice(len(xy), max_points, replace=False)
            xy, v_plot = xy[keep], v[keep]
        else:
            v_plot = v
        order = np.argsort(v_plot, kind="stable")                      # expressing cells on top
        prepared.append((p["title"], xy[order], v_plot[order], len(p["xy"]), float(np.mean(v))))
    allp = np.vstack([xy for _, xy, _, _, _ in prepared])
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.05 * (hi - lo + 1e-12)

    fig, axes = plt.subplots(1, len(prepared), figsize=(4.6 * len(prepared) + 1.0, 4.8), sharex=True, sharey=True,
                             squeeze=False)
    sc = None
    for ax, (name, xy, v, n, mean) in zip(axes[0], prepared):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=point_size, linewidths=0, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                        rasterized=True)
        ax.set_title(f"{name}\nn={n}, mean={mean:.3g}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=list(axes[0]), shrink=0.8, pad=0.02, label=gene)
    fig.suptitle(title)
    return fig


def run(a):
    import anndata as ad
    target = float(a.target)
    nb = check_times(target, a.neighbors)
    direction = None if len(nb) == 2 else "forward"
    os.makedirs(a.out, exist_ok=True)

    real = load_slices(a.original, a.time_key, a.spatial_key, nb + [target])
    var_names = list(real[target].var_names)
    kind = "interpolation" if direction is None else "forward extrapolation"
    written = []
    for gname in a.genes:
        gene = resolve_gene(gname, var_names)
        if gene is None:
            hint = suggest(gname, var_names)
            raise ValueError(f"gene '{gname}' not found in {a.original}" + (f"; similar names: {hint}" if hint else ""))
        obs_vals = {t: gene_values(real[t], gene) for t in nb + [target]}
        vlim = color_range(list(obs_vals.values()))

        def real_panel(t, label):
            return dict(title=label, xy=np.asarray(real[t].obsm[a.spatial_key]), vals=obs_vals[t])

        for model in a.models:
            name = common.pred_name(model, target, direction)
            path = os.path.join(a.pred_dir, name + ".h5ad")
            if not os.path.exists(path):
                print(f"[genes] {model}: no {name}.h5ad in {a.pred_dir}, skipped")
                continue
            pred = ad.read_h5ad(path)
            pg = resolve_gene(gene, list(pred.var_names))
            if pg is None:
                print(f"[genes] {model}: '{gene}' is not among its {pred.n_vars} predicted genes, skipped")
                continue
            panels = [real_panel(nb[0], f"t={nb[0]:g} observed"),
                      real_panel(target, f"t={target:g} TRUE"),
                      dict(title=f"t={target:g} PREDICTED ({model})", xy=np.asarray(pred.obsm[a.spatial_key]),
                           vals=gene_values(pred, pg))]
            if len(nb) == 2:
                panels.append(real_panel(nb[1], f"t={nb[1]:g} observed"))
            fig = build_figure(panels, gene, vlim, a.point_size, a.max_points, not a.no_center, a.cmap, a.seed,
                               title=f"{gene}: {kind} of t={target:g} with {model}")
            out = os.path.join(a.out, f"gene_{gene}__{name}.png")
            fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
            written.append(out)
            print(f"[genes] wrote {out}")
    return written


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--original", required=True, help="h5ad with every real timepoint (as given to the pipeline)")
    p.add_argument("--pred-dir", required=True, help="the predictions/ folder of a pipeline run")
    p.add_argument("--target", type=float, required=True, help="the predicted timepoint")
    p.add_argument("--neighbors", type=float, nargs="+", required=True,
                   help="two real timepoints around the target (interpolation) or one earlier one (extrapolation)")
    p.add_argument("--genes", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=list(common.MODELS), choices=list(common.MODELS))
    p.add_argument("--out", default="gene_plots")
    p.add_argument("--time-key", default="time")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--max-points", type=int, default=40000, help="draw at most this many cells per panel")
    p.add_argument("--no-center", action="store_true", help="raw coordinates instead of centring each panel")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
