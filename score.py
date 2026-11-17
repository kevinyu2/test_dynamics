#!/usr/bin/env python
"""
score.py -- score every prediction in a folder against the true slices of the full dataset.

    python score.py --pred-dir out/predictions --original full.h5ad --out out/scores

Reads the prediction files written by run_pipeline.py (model, target time and neighbouring
timepoints are stored inside each file). For every file it computes, against the true cells at the
target time, the metrics of interp_metrics.py:
    w2_expr, w2_joint, w2_joint_aligned (+ frit_angle_deg), gw_spatial, fgw
It also scores the two OBSERVED neighbouring slices against the truth ("observed_t<time>" rows):
what you would get with no interpolation at all.

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
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--original", required=True, help="the full dataset (all timepoints)")
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
    return p.parse_args(argv)


def _meta(pred, path):
    u = pred.uns
    try:
        return (str(u["interp_model"]), float(u["interp_target_time"]),
                [float(x) for x in np.asarray(u["interp_flank_times"]).ravel()],
                [float(x) for x in np.asarray(u["interp_support_times"]).ravel()])
    except KeyError as e:
        raise KeyError(f"{path} has no {e} -- was it written by run_pipeline.py?")


def _rows(model, target, support, res, file):
    base = dict(model=model, target_time=target, support_times=" ".join(f"{x:g}" for x in support),
                n_pred=res["n_pred"], n_true=res["n_true"], file=file)
    return [dict(base, metric=k, mean=v[0], std=v[1]) for k, v in res["pred"].items()]


def main(argv=None):
    a = parse_args(argv)
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
        model, target, flank, support = _meta(pred, path)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[score] {name}: {pred.n_obs} predicted cells vs true t={target:g}")
        res = evaluate_interpolation(pred, original, target, pred_spatial_key="spatial", baseline_times=(), **kw)
        json.dump(res, open(os.path.join(a.out, "json", f"{name}.json"), "w"), indent=1)
        long_rows += _rows(model, target, support, res, os.path.basename(path))

        if not a.no_baselines:                                  # observed neighbours vs the same truth
            for ft in flank:
                if (target, ft) in done_baselines:
                    continue
                done_baselines.add((target, ft))
                tt = np.asarray(original.obs[a.time_key], dtype=float)
                nb = original[np.isclose(tt, ft)]
                bres = evaluate_interpolation(nb, original, target, pred_spatial_key=a.spatial_key, baseline_times=(), **kw)
                long_rows += _rows(f"observed_t{ft:g}", target, [ft], bres, "")

    long = pd.DataFrame(long_rows)
    long.to_csv(os.path.join(a.out, "scores_long.csv"), index=False)
    keys = ["model", "target_time", "support_times", "n_pred", "n_true", "file"]
    mean = long.pivot_table(index=keys, columns="metric", values="mean", sort=False)
    std = long.pivot_table(index=keys, columns="metric", values="std", sort=False).add_suffix("_std")
    wide = pd.concat([mean, std], axis=1).reset_index()
    wide.to_csv(os.path.join(a.out, "scores_wide.csv"), index=False)

    pd.set_option("display.width", 200)
    for metric in [m for m in mean.columns]:
        t = long[long.metric == metric].pivot_table(index="model", columns="target_time", values="mean", sort=False)
        note = "fitted rotation of the true slice, a diagnostic only" if metric == "frit_angle_deg" else "lower is better"
        print(f"\n== {metric} (rows: model, columns: held-out time; {note})")
        print(t.round(4).to_string())
    print(f"\n[score] wrote {a.out}/scores_wide.csv, scores_long.csv and json/")
    return wide


if __name__ == "__main__":
    main()