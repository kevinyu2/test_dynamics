"""
common.py -- helpers shared by call_moscot.py, call_stvcr.py, run_pipeline.py and score.py.

Dataset format expected everywhere (same as the toy dataset):
    adata.X                  expression (log-normalised), cells x genes, ideally scipy sparse
    adata.obs['time']        float timepoint of each cell
    adata.obs['Annotation']  cell type (categorical)            [name configurable]
    adata.obsm['spatial']    (n_cells, 2 or 3) coordinates      [name configurable]
    adata.uns['Annotation_colors']  optional list of colours, one per category

Every prediction file written by run_pipeline.py is an AnnData with
    X, var_names        predicted expression
    obsm['spatial']     predicted coordinates, in raw coordinate UNITS (see run_pipeline.py)
    obs[label_key]      predicted cell type
    uns['interp_*']     metadata: model, target time, support times, flanking times, parameters
"""
import json

import numpy as np
import pandas as pd

MODELS = ("moscot", "stvcr_all", "stvcr_two")


# ----------------------------------------------------------------- small utils
def time_tag(t):
    """2.0 -> '2', 2.5 -> '2p5' (safe inside file names)."""
    return f"{float(t):g}".replace(".", "p").replace("-", "m")


DIRECTION_TAG = {"forward": "fwd_"}


def pred_name(model, target, direction=None):
    """moscot__t2 for interpolation; moscot__fwd_t4 for forward extrapolation."""
    return f"{model}__{DIRECTION_TAG.get(direction, '')}t{time_tag(target)}"


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------ choosing the data
def select_times(adata, time_key, times=None, n_cells=None, seed=0):
    """
    Restrict the dataset to some timepoints and/or subsample cells per timepoint.
    Returns (copy of the subset, sorted array of the timepoints kept).
    """
    t = np.asarray(adata.obs[time_key], dtype=float)
    all_times = np.unique(t)
    if times is None:
        keep = all_times
    else:
        keep = []
        for want in times:
            hit = all_times[np.isclose(all_times, float(want))]
            if len(hit) == 0:
                raise ValueError(f"timepoint {want} is not in the data; available: {all_times.tolist()}")
            keep.append(hit[0])
        keep = np.array(sorted(set(keep)))

    rng = np.random.default_rng(seed)
    idx = []
    for k in keep:
        ii = np.where(t == k)[0]
        if n_cells is not None and len(ii) > n_cells:
            ii = rng.choice(ii, n_cells, replace=False)
        idx.append(ii)
    idx = np.sort(np.concatenate(idx))
    return adata[idx].copy(), keep


def plan_runs(times, targets=None, models=MODELS, mode="both"):
    """
    Decide what to predict and from which timepoints.

    mode="both" (default) plans the interpolation runs and then the forward-extrapolation runs (see _plan_both).
    mode="extrapolate" predicts timepoints OUTSIDE the data the model sees (see _plan_extrapolation).
    The rest of this docstring describes mode="interpolate".

    times    observed timepoints that are available (after any --times restriction)
    targets  intermediate timepoints to hold out and predict (default: every one that has an
             observed timepoint on both sides, i.e. all except the first and last)
    models   subset of MODELS

    For each target, with left/right = nearest available timepoints before/after it:
      moscot     support = [left, right]
      stvcr_two  support = [left, right]
      stvcr_all  support = every available timepoint except the target
    Returns a list of dicts: model, target, support, flank (+ 'skip' reason if redundant).
    """
    times = sorted(float(x) for x in times)
    if mode == "both":
        return _plan_both(times, targets, models)
    if mode == "extrapolate":
        return _plan_extrapolation(times, targets, models)
    if mode != "interpolate":
        raise ValueError(f"mode must be 'both', 'interpolate' or 'extrapolate', got {mode!r}")
    if len(times) < 3:
        raise ValueError("need at least 3 timepoints: one to hold out and one on each side")
    inner = times[1:-1]
    if targets is None:
        targets = inner
    else:
        chosen = []
        for want in targets:
            hit = [t for t in times if np.isclose(t, float(want))]
            if not hit:
                raise ValueError(f"target {want} is not among the timepoints {times}")
            if hit[0] not in inner:
                raise ValueError(f"target {want} is the first or last timepoint, so it has no timepoint on one side")
            chosen.append(hit[0])
        targets = sorted(set(chosen))

    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; choose from {MODELS}")

    runs = []
    for m in models:
        for tgt in targets:
            left = max(t for t in times if t < tgt)
            right = min(t for t in times if t > tgt)
            others = [t for t in times if t != tgt]
            support = others if m == "stvcr_all" else [left, right]
            run = dict(model=m, target=tgt, support=support, flank=[left, right], mode="interpolate", direction=None)
            if m == "stvcr_two" and support == others:
                run["skip"] = "same training timepoints as stvcr_all"
            runs.append(run)
    return runs


def _plan_both(times, targets, models):
    """
    Interpolation runs first, then forward-extrapolation runs.

    Without `targets`: every interpolatable timepoint (all but the first and last) and every extrapolatable one
    (all but the first two). With `targets`: each target is used for every mode it is valid for, and it is an
    error only if it is valid for neither (e.g. the first timepoint) or is not a timepoint at all. So with
    times 1..4 and --targets 3 4, timepoint 3 is both interpolated and extrapolated, 4 only extrapolated.
    """
    if len(times) < 3:
        raise ValueError("need at least 3 timepoints: one to hold out and one or two around it")
    interp_ok, extrap_ok = times[1:-1], times[2:]
    if targets is None:
        t_int, t_ext = list(interp_ok), list(extrap_ok)
    else:
        t_int, t_ext = [], []
        for want in targets:
            hit = [t for t in times if np.isclose(t, float(want))]
            if not hit:
                raise ValueError(f"target {want} is not among the timepoints {times}")
            if hit[0] not in interp_ok and hit[0] not in extrap_ok:
                raise ValueError(f"target {want:g} can be neither interpolated nor extrapolated forward "
                                 f"(interpolation targets {interp_ok}, extrapolation targets {extrap_ok})")
            if hit[0] in interp_ok:
                t_int.append(hit[0])
            if hit[0] in extrap_ok:
                t_ext.append(hit[0])
    runs = []
    if t_int:
        runs += plan_runs(times, sorted(set(t_int)), models, mode="interpolate")
    if t_ext:
        runs += plan_runs(times, sorted(set(t_ext)), models, mode="extrapolate")
    return runs


def _plan_extrapolation(times, targets, models):
    """
    Hold out a timepoint and predict it FORWARD in time from the timepoints before it (at least two needed).

    For each target, with `side` = all timepoints before it:
      moscot     support = the two latest timepoints of `side`
      stvcr_two  support = the same two
      stvcr_all  support = all of `side`
    flank = [the single nearest earlier timepoint]: the naive baseline is "no change from the nearest slice".
    Later timepoints are NOT used at all (that would be interpolation). There is no backward extrapolation.
    """
    if len(times) < 3:
        raise ValueError("need at least 3 timepoints: one to hold out and two before it")
    bad = [m for m in models if m not in MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; choose from {MODELS}")
    valid = times[2:]                                                 # targets with >= 2 earlier timepoints

    if targets is None:
        chosen = list(valid)
    else:
        chosen = []
        for want in targets:
            hit = [t for t in times if np.isclose(t, float(want))]
            if not hit:
                raise ValueError(f"target {want} is not among the timepoints {times}")
            if hit[0] not in valid:
                raise ValueError(f"target {want:g} cannot be extrapolated forward: it needs at least two earlier "
                                 f"timepoints (valid targets {valid})")
            chosen.append(hit[0])
        chosen = sorted(set(chosen))

    runs = []
    for m in models:
        for tgt in chosen:
            side = [t for t in times if t < tgt]
            support = side if m == "stvcr_all" else side[-2:]
            run = dict(model=m, target=tgt, support=support, flank=[side[-1]], mode="extrapolate", direction="forward")
            if m == "stvcr_two" and support == side:
                run["skip"] = "same training timepoints as stvcr_all"
            runs.append(run)
    return runs


# ---------------------------------------------------------------------- colours
def label_colors(adata, label_key):
    """Ordered dict {category: colour}. Uses adata.uns[label_key + '_colors'] if it fits."""
    cats = list(pd.Categorical(adata.obs[label_key]).categories)
    cols = adata.uns.get(f"{label_key}_colors")
    if cols is not None and len(list(cols)) >= len(cats):
        cols = [str(c) for c in list(cols)[: len(cats)]]
    else:
        import matplotlib
        from matplotlib.colors import to_hex
        cmap = matplotlib.colormaps["tab20"]
        cols = [to_hex(cmap(i % 20)) for i in range(len(cats))]
    return dict(zip([str(c) for c in cats], cols))


def finalize_prediction(pred, *, model, target, support, flank, label_key, colors, params,
                        mode="interpolate", direction=None):
    """Put a model's raw output into the common on-disk format (see module docstring)."""
    cats = list(colors)
    if label_key in pred.obs:
        pred.obs[label_key] = pd.Categorical(pred.obs[label_key].astype(str).values, categories=cats)
    pred.uns[f"{label_key}_colors"] = [colors[c] for c in cats]          # a real list: h5ad can't store dict_values
    pred.uns["interp_model"] = str(model)
    pred.uns["interp_target_time"] = float(target)
    pred.uns["interp_support_times"] = np.asarray(support, dtype=float)
    pred.uns["interp_flank_times"] = np.asarray(flank, dtype=float)
    pred.uns["interp_label_key"] = str(label_key)
    pred.uns["interp_mode"] = str(mode)                                  # "interpolate" or "extrapolate"
    pred.uns["interp_direction"] = str(direction or "")                  # "" or "forward"
    pred.uns["interp_params_json"] = json.dumps(params, default=_jsonable)
    pred.obs["time"] = float(target)
    return pred


# ------------------------------------------------------------------------ plots
def plot_prediction(pred_xy, pred_lab, truth_xy, truth_lab, colors, path, title="", center=True):
    """
    Cell-type scatter of a predicted slice next to the true held-out slice.
    Both panels are centred on their own centroid (stVCR works in a centred frame; if the real
    slices differ by a rotation the two panels can differ by that rotation too).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [("predicted", pred_xy, pred_lab)]
    if truth_xy is not None:
        panels.append(("true (held out)", truth_xy, truth_lab))
    pts = []
    for _, xy, _ in panels:
        xy = np.asarray(xy, dtype=float)[:, :2]
        pts.append(xy - xy.mean(0) if center else xy)
    allp = np.vstack(pts)
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.05 * (hi - lo + 1e-12)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5), sharex=True, sharey=True, squeeze=False)
    for ax, (name, _, lab), xy in zip(axes[0], panels, pts):
        ax.scatter(xy[:, 0], xy[:, 1], c=[colors.get(str(l), "#999999") for l in lab], s=6, linewidths=0)
        ax.set_title(f"{name} (n={len(xy)})")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c, markeredgecolor=c, markersize=7, label=str(k))
               for k, c in colors.items()]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, title="cell type")
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_adata(pred, truth, path, label_key, colors, title="", spatial_key="spatial"):
    return plot_prediction(
        np.asarray(pred.obsm[spatial_key]), pred.obs[label_key].astype(str).values,
        None if truth is None else np.asarray(truth.obsm[spatial_key]),
        None if truth is None else truth.obs[label_key].astype(str).values,
        colors, path, title)