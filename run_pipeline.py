#!/usr/bin/env python
"""
run_pipeline.py -- hold out each intermediate timepoint and predict it with three setups.

  moscot      uses only the two nearest observed timepoints (one before, one after the target)
  stvcr_two   stVCR trained on those same two timepoints
  stvcr_all   stVCR trained on ALL the other timepoints (everything except the target)

Input : ONE h5ad containing every timepoint (format described in common.py).
Output: <out>/
          predictions/<model>__t<time>.h5ad   one file per (model, target) -> feed this folder to score.py
          images/<model>__t<time>.png         predicted cell types next to the true slice
          runs/<model>__t<time>/              stVCR model files (+ stVCR's own plot)
          manifest.csv, config.json, errors.log

Choosing what to run (all optional):
  --times 1 2 4 5     use only these observed timepoints; the rest of the data is dropped first
                      (so with --times 1 3 5 the "nearest" timepoints of 3 are 1 and 5)
  --targets 3         hold out and predict only these (default: every timepoint with data on both sides)
  --models moscot     run only some of moscot / stvcr_two / stvcr_all
  --n-cells 500       random subsample per timepoint (fast tests)
  --quick             short stVCR training (epochs 201, 300 samples per step, AE 100 epochs)

Extrapolation instead of interpolation:
  --mode extrapolate  hold out a timepoint and predict it FORWARD in time from the timepoints before it
                      (needs >= 2 earlier timepoints, e.g. 1 2 3 -> 4); there is no backward extrapolation.
      moscot / stvcr_two use the two nearest earlier timepoints, stvcr_all uses all earlier ones,
      and score.py's naive baseline is the single nearest observed slice. Files are named
      <model>__fwd_t<time>.h5ad. --targets picks which times to predict (default: every valid one).

Then:  python score.py --pred-dir <out>/predictions --original <same input file> --out <out>/scores
"""
import argparse
import json
import os
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

import common
from common import MODELS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="h5ad with ALL timepoints")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--time-key", default="time")
    p.add_argument("--spatial-key", default="spatial")
    p.add_argument("--label-key", default="Annotation")
    p.add_argument("--times", type=float, nargs="*", default=None, help="only use these observed timepoints")
    p.add_argument("--targets", type=float, nargs="*", default=None, help="only hold out / predict these")
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--mode", default="interpolate", choices=["interpolate", "extrapolate"],
                   help="interpolate: hold out inner timepoints; extrapolate: predict beyond the data")
    p.add_argument("--n-cells", type=int, default=None, help="subsample this many cells per timepoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="redo predictions that already exist")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first error (default: log it and go on)")
    p.add_argument("--quick", action="store_true", help="short stVCR training for testing")
    # moscot
    p.add_argument("--moscot-tau", type=float, default=0.9, help="1.0 = balanced; <1 allows growth/death")
    p.add_argument("--moscot-epsilon", type=float, default=1e-2)
    p.add_argument("--moscot-alpha", type=float, default=0.5)
    p.add_argument("--n-pca", type=int, default=10)
    p.add_argument("--moscot-jitter", type=float, default=0.0)
    # stVCR
    p.add_argument("--stvcr-epochs", type=int, default=None, help="default 2001 (201 with --quick)")
    p.add_argument("--stvcr-num-samples", type=int, default=None, help="default 1000 (300 with --quick)")
    p.add_argument("--ae-epochs", type=int, default=None, help="default 1000 (100 with --quick)")
    p.add_argument("--z-dims", type=int, default=10)
    p.add_argument("--delta-t", type=float, default=0.1, help="Euler step (time units) for stVCR interpolation")
    p.add_argument("--time-unit", default="day")
    p.add_argument("--no-initial-alignment", action="store_true")
    p.add_argument("--stvcr-min-genes", type=int, default=100)
    p.add_argument("--stvcr-min-cells", type=int, default=3)
    p.add_argument("--stvcr-n-hvg", type=int, default=2000)
    a = p.parse_args(argv)
    a.stvcr_epochs = a.stvcr_epochs or (201 if a.quick else 2001)
    a.stvcr_num_samples = a.stvcr_num_samples or (300 if a.quick else 1000)
    a.ae_epochs = a.ae_epochs or (100 if a.quick else 1000)
    return a


def run_one(run, adata, args, colors, out):
    """Train one model on its support timepoints and return its raw (unfinalised) prediction + params."""
    t = np.asarray(adata.obs[args.time_key], dtype=float)
    support = adata[np.isin(t, run["support"])].copy()
    name = common.pred_name(run["model"], run["target"], run.get("direction"))
    extrapolating = run.get("mode") == "extrapolate"
    if run["model"] == "moscot":
        from call_moscot import interpolate_moscot, train_moscot
        params = dict(n_pca=args.n_pca, epsilon=args.moscot_epsilon, alpha=args.moscot_alpha, tau=args.moscot_tau,
                      jitter=args.moscot_jitter, seed=args.seed)
        model = train_moscot(support, args.time_key, args.spatial_key, args.n_pca, args.moscot_epsilon,
                             args.moscot_alpha, args.moscot_tau, args.label_key)
        pred = interpolate_moscot(model, run["target"], jitter=args.moscot_jitter, seed=args.seed,
                                  allow_extrapolation=extrapolating)
    else:
        from call_stvcr import run_stvcr
        params = dict(n_epochs=args.stvcr_epochs, num_samples=args.stvcr_num_samples, ae_epochs=args.ae_epochs,
                      z_dims=args.z_dims, delta_t=args.delta_t, seed=args.seed,
                      use_initial_alignment=not args.no_initial_alignment)
        pred = run_stvcr(support, run["target"], os.path.join(out, "runs", name), args.time_key, args.spatial_key,
                         args.label_key, colors, args.stvcr_min_genes, args.stvcr_min_cells, args.stvcr_n_hvg,
                         args.stvcr_epochs, args.stvcr_num_samples, args.z_dims, args.ae_epochs,
                         not args.no_initial_alignment, args.delta_t, args.time_unit, seed=args.seed)
    return pred, params


def main(argv=None):
    args = parse_args(argv)
    import anndata as ad

    out = os.path.abspath(args.out)
    for sub in ("predictions", "images", "runs"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    full = ad.read_h5ad(args.input)
    for key, where in ((args.time_key, full.obs), (args.label_key, full.obs), (args.spatial_key, full.obsm)):
        if key not in where:
            raise KeyError(f"'{key}' not found in the input file (obs: {list(full.obs.columns)}, obsm: {list(full.obsm.keys())})")
    colors = common.label_colors(full, args.label_key)
    adata, times = common.select_times(full, args.time_key, args.times, args.n_cells, args.seed)
    runs = common.plan_runs(times, args.targets, args.models, mode=args.mode)
    targets = sorted({r["target"] for r in runs})
    print(f"[pipeline] {adata.n_obs} cells, timepoints {times.tolist()}, targets {targets}, models {args.models}, "
          f"mode {args.mode}" + (" (forward)" if args.mode == "extrapolate" else ""))

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "timepoints": times.tolist(), "targets": targets,
                   "n_cells": int(adata.n_obs), "runs": runs}, f, indent=1, default=common._jsonable)

    t_all = np.asarray(adata.obs[args.time_key], dtype=float)
    manifest_path = os.path.join(out, "manifest.csv")
    old_all = {}
    if os.path.exists(manifest_path):                                  # keep the record of earlier invocations
        old_all = {r["file"]: r for r in pd.read_csv(manifest_path).fillna("").to_dict("records")}
    old = {f: r for f, r in old_all.items() if r["status"] in ("ok", "exists")}

    def write_manifest(rows):
        merged = dict(old_all)
        merged.update({r["file"]: r for r in rows})
        pd.DataFrame(list(merged.values())).to_csv(manifest_path, index=False)
    rows = []
    for i, run in enumerate(runs, 1):
        name = common.pred_name(run["model"], run["target"], run.get("direction"))
        row = dict(model=run["model"], target_time=run["target"], support_times=" ".join(f"{x:g}" for x in run["support"]),
                   mode=run.get("mode", "interpolate"), direction=run.get("direction") or "",
                   file=f"predictions/{name}.h5ad", image=f"images/{name}.png", status="", seconds=0.0, n_pred=0, message="")
        dst = os.path.join(out, "predictions", f"{name}.h5ad")
        print(f"\n[pipeline] ({i}/{len(runs)}) {name}: train on {run['support']} -> predict {run['target']:g}")
        if "skip" in run:
            row.update(status="skipped", message=run["skip"]); rows.append(row); print("   skipped:", run["skip"]); continue
        if os.path.exists(dst) and not args.overwrite:
            row = dict(old.get(row["file"], row), status="ok" if row["file"] in old else "exists")
            rows.append(row); print("   already there (use --overwrite to redo)"); continue
        t0 = time.time()
        try:
            pred, params = run_one(run, adata, args, colors, out)
            pred = common.finalize_prediction(pred, model=run["model"], target=run["target"], support=run["support"],
                                              flank=run["flank"], label_key=args.label_key, colors=colors, params=params,
                                              mode=run.get("mode", "interpolate"), direction=run.get("direction"))
            pred.write_h5ad(dst)
            truth = adata[np.isclose(t_all, run["target"])]
            how = " (forward extrapolation)" if run.get("direction") == "forward" else ""
            title = f"{run['model']} | predict t={run['target']:g} from t={row['support_times']}{how}"
            common.plot_prediction_adata(pred, truth, os.path.join(out, row["image"]), args.label_key, colors, title)
            row.update(status="ok", n_pred=int(pred.n_obs))
        except Exception as e:                                         # keep going: one failure shouldn't lose the rest
            row.update(status="failed", message=f"{type(e).__name__}: {e}")
            with open(os.path.join(out, "errors.log"), "a") as f:
                f.write(f"== {name}\n{traceback.format_exc()}\n")
            print("   FAILED:", row["message"])
            if args.fail_fast:
                raise
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        write_manifest(rows)

    write_manifest(rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n[pipeline] done: {n_ok} ok, {sum(r['status'] == 'failed' for r in rows)} failed. Next:")
    print(f"  python score.py --pred-dir {out}/predictions --original {args.input} --out {out}/scores")
    return rows


if __name__ == "__main__":
    main()