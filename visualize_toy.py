#!/usr/bin/env python
"""
visualize_toy.py -- one-page overview of a dataset in the toy format (works for any such h5ad).

    python visualize_toy.py --input toy_all.h5ad --out toy_overview.png

Top row    : cells in space at every timepoint, coloured by cell type (shared axes)
Bottom row : cell-type proportions over time | expression PCA by cell type | expression PCA by time
"""
import argparse

import numpy as np


def plot_overview(xy, time, labels, expr, colors, out, title="dataset overview", max_cols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xy, time, labels = np.asarray(xy, float), np.asarray(time, float), np.asarray(labels).astype(str)
    times = np.unique(time)
    cats = list(colors)
    ncol = max(len(times), 3)
    fig = plt.figure(figsize=(3.2 * ncol, 7.4))
    gs = fig.add_gridspec(2, ncol, hspace=0.35, wspace=0.4)

    lo, hi = xy.min(0), xy.max(0)
    pad = 0.05 * (hi - lo)
    for j, t in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        m = time == t
        ax.scatter(xy[m, 0], xy[m, 1], c=[colors.get(l, "#999999") for l in labels[m]], s=4, linewidths=0)
        ax.set_title(f"t = {t:g}  (n={m.sum()})", fontsize=10)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    bottom = np.zeros(len(times))
    for c in cats:
        frac = np.array([(labels[time == t] == c).mean() for t in times])
        ax.bar(range(len(times)), frac, bottom=bottom, color=colors[c], label=c)
        bottom += frac
    ax.set_xticks(range(len(times))); ax.set_xticklabels([f"{t:g}" for t in times])
    ax.set_title("cell-type proportions", fontsize=10); ax.set_xlabel("time"); ax.set_ylim(0, 1)

    Xc = np.asarray(expr, float) - np.asarray(expr, float).mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(pcs[:, 0], pcs[:, 1], c=[colors.get(l, "#999999") for l in labels], s=3, linewidths=0)
    ax.set_title("expression PCA: cell type", fontsize=10); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax = fig.add_subplot(gs[1, 2])
    sc_ = ax.scatter(pcs[:, 0], pcs[:, 1], c=time, cmap="viridis", s=3, linewidths=0)
    ax.set_title("expression PCA: time", fontsize=10); ax.set_xlabel("PC1")
    fig.colorbar(sc_, ax=ax, fraction=0.046)

    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[c], markeredgecolor=colors[c], label=c)
               for c in cats]
    fig.legend(handles=handles, loc="center right", frameon=False, title="cell type")
    fig.suptitle(title)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="toy_all.h5ad")
    ap.add_argument("--out", default="toy_overview.png")
    ap.add_argument("--time-key", default="time")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--label-key", default="Annotation")
    a = ap.parse_args()

    import anndata as ad
    from common import dense, label_colors
    adata = ad.read_h5ad(a.input)
    colors = label_colors(adata, a.label_key)
    print(adata)
    print(adata.obs.groupby(a.time_key).size().to_string())
    plot_overview(np.asarray(adata.obsm[a.spatial_key])[:, :2], adata.obs[a.time_key].values,
                  adata.obs[a.label_key].astype(str).values, dense(adata.X), colors, a.out, title=a.input)
    print("wrote", a.out)


if __name__ == "__main__":
    main()