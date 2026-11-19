#!/usr/bin/env python
"""
score.py -- score every prediction in a folder against the true slices of the full dataset.

    python score.py --pred-dir out/predictions --original full.h5ad --out out/scores

Reads the prediction files written by run_pipeline.py (model, target time and neighbouring
timepoints are stored inside each file). For every file it computes, against the true cells at the
target time, the metrics of interp_metrics.py:
    w2_expr, w2_joint, w2_joint_aligned (+ frit_angle_deg), gw_spatial, fgw
It also scores the OBSERVED neighbouring slices against the truth ("observed_t<time>" rows):
what you would get with no interpolation at all. For extrapolation runs (run_pipeline.py --mode extrapolate)
the baseline is the single nearest observed slice, and every row carries a `task` column:
interp or extrap_fwd.

Fairness choices (all switchable):
  * the same gene list for every model (the genes all predictions share with the original), because
    stVCR predicts only its highly-variable genes while moscot predicts all of them;
  * every slice is centred on its own centroid (--no-center to turn off); predictions are already in
    raw coordinate units, so no rescaling is needed.

Writes into --out:
    scores_wide.csv   one row per (model, target): metric means, and <metric>_std over the repeats
    scores_long.csv   the same in long format (model, target, metric, mean, std)
    json/<name>.json  the full result of each evaluation
and prints a model x target table for each metric.
It also draws bar plots into --out/plots/: one PNG per metric (groups = held-out time, or forward
extrapolation target, bars = models, error bars = std over repeats) plus all_metrics.png. Replot without rescoring:
    python score.py --plot-only --out out/scores
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from interp_metrics import evaluate_interpolation


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", help="folder of prediction .h5ad files")
    p.add_argument("--original", help="the full dataset (all timepoints)")
    p.add_argument("--out", required=True)
    p.add_argument("--time-key", default="time")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--expr-key", default=None, help="obsm key instead of gene space (must exist in BOTH)")
    p.add_argument("--w", type=float, default=0.9, help="spatial weight of the joint W2 (paper: 0.9)")
    p.add_argument("--alpha", type=float, default=0.5, help="structure weight of FGW (extra metric)")
    p.add_argument("--n-max", type=int, default=1000, help="subsample size per slice (GW/FRIT cost grows fast)")
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-center", action="store_true")
    p.add_argument("--no-gw", action="store_true", help="skip gw_spatial and fgw")
    p.add_argument("--no-frit", action="store_true", help="skip w2_joint_aligned")
    p.add_argument("--own-genes", action="store_true", help="use each file's own gene overlap instead of a common list")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="do not draw the bar plots")
    p.add_argument("--plot-only", action="store_true", help="just redraw plots from <out>/scores_long.csv")
    a = p.parse_args(argv)
    if not a.plot_only and not (a.pred_dir and a.original):
        p.error("--pred-dir and --original are required (unless --plot-only)")
    return a


TASK_FOR_DIRECTION = {"forward": "extrap_fwd"}


def _meta(pred, path):
    """-> (model, target, flank times, support times, task). Files from before extrapolation existed count as interp."""
    u = pred.uns
    try:
        task = "interp"
        if str(u.get("interp_mode", "interpolate")) == "extrapolate":
            task = TASK_FOR_DIRECTION.get(str(u.get("interp_direction", "")), "extrap")
        return (str(u["interp_model"]), float(u["interp_target_time"]),
                [float(x) for x in np.asarray(u["interp_flank_times"]).ravel()],
                [float(x) for x in np.asarray(u["interp_support_times"]).ravel()], task)
    except KeyError as e:
        raise KeyError(f"{path} has no {e} -- was it written by run_pipeline.py?")


def _rows(model, target, support, res, file, task="interp"):
    base = dict(model=model, task=task, target_time=target, support_times=" ".join(f"{x:g}" for x in support),
                n_pred=res["n_pred"], n_true=res["n_true"], file=file)
    return [dict(base, metric=k, mean=v[0], std=v[1]) for k, v in res["pred"].items()]


# ------------------------------------------------------------------------------------------ plots
MODEL_ORDER = ["moscot", "stvcr_two", "stvcr_all"]
MODEL_COLORS = {"moscot": "#4C78A8", "stvcr_two": "#F58518", "stvcr_all": "#54A24B", "observed": "#B0B0B0"}
METRIC_NOTES = {"w2_expr": "expression W2", "w2_joint": "joint W2", "w2_joint_aligned": "joint W2 (aligned)",
                "gw_spatial": "spatial GW", "fgw": "FGW"}


TASK_ORDER = ["interp", "extrap_fwd"]
GROUP_LABEL = {"interp": "held-out {t}", "extrap_fwd": "forward {t}"}


def _bar_data(long, metric, baselines=True):
    """-> (groups, series, obs_label). groups = [(task, target_time)] in plot order; series = {name: (means, stds)}
    aligned to groups. Observed neighbours are collapsed to the better one per group (one grey bar)."""
    d = long[long.metric == metric]
    groups = sorted({(r.task, r.target_time) for r in d.itertuples()},
                    key=lambda g: (TASK_ORDER.index(g[0]) if g[0] in TASK_ORDER else 99, g[1]))
    is_obs = d.model.astype(str).str.startswith("observed")
    models = [m for m in MODEL_ORDER if m in set(d.model)] + sorted(set(d.model[~is_obs]) - set(MODEL_ORDER))

    def pick(rows, g):
        r = rows[(rows.task == g[0]) & (rows.target_time == g[1])]
        return r

    series = {}
    for m in models:
        r = d[d.model == m]
        vals = [pick(r, g) for g in groups]
        series[m] = ([v["mean"].iloc[0] if len(v) else np.nan for v in vals],
                     [v["std"].iloc[0] if len(v) else np.nan for v in vals])
    obs_label = "observed neighbour"
    obs = d[is_obs]
    if baselines and len(obs):
        means, stds, counts = [], [], []
        for g in groups:
            o = pick(obs, g)
            counts.append(len(o))
            if len(o):
                best = o.loc[o["mean"].idxmin()]
                means.append(best["mean"]); stds.append(best["std"])
            else:
                means.append(np.nan); stds.append(np.nan)
        series["observed"] = (means, stds)
        nz = [c for c in counts if c]
        obs_label = ("nearest observed slice" if max(nz) == 1 else
                     "observed neighbour (better of 2)" if min(nz) == 2 else "observed neighbour (best available)")
    return groups, series, obs_label


def _draw_bars(ax, groups, series, title, obs_label="observed neighbour"):
    labels = {"observed": obs_label}
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for i, (name, (means, stds)) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means, width * 0.92, yerr=np.nan_to_num(stds), capsize=3, label=labels.get(name, name),
               color=MODEL_COLORS.get(name, None), edgecolor="black", linewidth=0.6,
               hatch="//" if name == "observed" else None, error_kw=dict(lw=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_LABEL.get(task, task + " {t}").format(t=f"{t:g}") for task, t in groups])
    ax.set_title(title)
    ax.set_ylabel("lower is better")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_scores(long, out_dir, baselines=True):
    """One bar plot per metric (groups = held-out time, bars = models) plus an overview grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if "task" not in long.columns:                                      # scores_long.csv from before extrapolation existed
        long = long.assign(task="interp")
    metrics = [m for m in long.metric.unique() if m != "frit_angle_deg"]      # the angle is a diagnostic, not a score
    paths = []
    for m in metrics:
        groups, series, obs_label = _bar_data(long, m, baselines)
        fig, ax = plt.subplots(figsize=(max(4.5, 1.9 * len(groups) + 2), 4))
        _draw_bars(ax, groups, series, METRIC_NOTES.get(m, m), obs_label)
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        p = os.path.join(out_dir, f"{m}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    if metrics:
        ncol = min(3, len(metrics)); nrow = int(np.ceil(len(metrics) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.8 * nrow), squeeze=False)
        for ax, m in zip(axes.ravel(), metrics):
            groups, series, obs_label = _bar_data(long, m, baselines)
            _draw_bars(ax, groups, series, METRIC_NOTES.get(m, m), obs_label)
        for ax in axes.ravel()[len(metrics):]:
            ax.axis("off")
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p = os.path.join(out_dir, "all_metrics.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv=None):
    a = parse_args(argv)
    if a.plot_only:
        paths = plot_scores(pd.read_csv(os.path.join(a.out, "scores_long.csv")), os.path.join(a.out, "plots"),
                            baselines=not a.no_baselines)
        print("[score] wrote", *paths, sep="\n  ")
        return
    import anndata as ad

    os.makedirs(os.path.join(a.out, "json"), exist_ok=True)
    original = ad.read_h5ad(a.original)
    files = sorted(glob.glob(os.path.join(a.pred_dir, "*.h5ad")))
    if not files:
        raise SystemExit(f"no .h5ad files in {a.pred_dir}")

    preds = [(f, ad.read_h5ad(f)) for f in files]
    common_genes = None
    if a.expr_key is None and not a.own_genes:
        shared = set(original.var_names)
        for _, p in preds:
            shared &= set(p.var_names)
        common_genes = [g for g in original.var_names if g in shared]
        print(f"[score] {len(common_genes)} genes shared by the original and all {len(preds)} predictions")
        if not common_genes:
            raise SystemExit("no gene names are shared by all predictions; check pred.var_names (see call_stvcr.py)")

    kw = dict(time_key=a.time_key, spatial_key=a.spatial_key, expr_key=a.expr_key, truth_spatial_scale=1.0,
              alpha=a.alpha, n_max=a.n_max, n_repeats=a.n_repeats, seed=a.seed, w=a.w, center=not a.no_center,
              gw=not a.no_gw, frit=not a.no_frit, genes=common_genes)
    long_rows, done_baselines = [], set()
    for path, pred in preds:
        model, target, flank, support, task = _meta(pred, path)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[score] {name}: {pred.n_obs} predicted cells vs true t={target:g} ({task})")
        res = evaluate_interpolation(pred, original, target, pred_spatial_key="spatial", baseline_times=(), **kw)
        json.dump(res, open(os.path.join(a.out, "json", f"{name}.json"), "w"), indent=1)
        long_rows += _rows(model, target, support, res, os.path.basename(path), task)

        if not a.no_baselines:                                  # observed neighbours vs the same truth
            for ft in flank:
                if (task, target, ft) in done_baselines:
                    continue
                done_baselines.add((task, target, ft))
                tt = np.asarray(original.obs[a.time_key], dtype=float)
                nb = original[np.isclose(tt, ft)]
                bres = evaluate_interpolation(nb, original, target, pred_spatial_key=a.spatial_key, baseline_times=(), **kw)
                long_rows += _rows(f"observed_t{ft:g}", target, [ft], bres, "", task)

    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(a.out, "scores_long.csv"), index=False)
    keys = ["model", "task", "target_time", "support_times", "n_pred", "n_true", "file"]
    mean = long.pivot_table(index=keys, columns="metric", values="mean", sort=False)
    std = long.pivot_table(index=keys, columns="metric", values="std", sort=False).add_suffix("_std")
    wide = pd.concat([mean, std], axis=1).reset_index()
    wide.to_csv(os.path.join(a.out, "scores_wide.csv"), index=False)

    pd.set_option("display.width", 200)
    for metric in [m for m in mean.columns]:
        t = long[long.metric == metric].pivot_table(index="model", columns=["task", "target_time"], values="mean", sort=False)
        note = "fitted rotation of the true slice, a diagnostic only" if metric == "frit_angle_deg" else "lower is better"
        print(f"\n== {metric} (rows: model, columns: task / held-out time; {note})")
        print(t.round(4).to_string())
    print(f"\n[score] wrote {a.out}/scores_wide.csv, scores_long.csv and json/")
    if not a.no_plots:
        paths = plot_scores(long, os.path.join(a.out, "plots"), baselines=not a.no_baselines)
        print(f"[score] wrote {len(paths)} plots to {a.out}/plots/")
    return wide


if __name__ == "__main__":
    main()