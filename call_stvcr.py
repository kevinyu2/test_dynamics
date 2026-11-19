"""
call_stvcr.py -- train stVCR on a set of timepoints and interpolate a held-out timepoint.

    data  = prepare_stvcr_data(support_adata, ...)        # scanpy filtering, stVCR-friendly copy
    model = train_stvcr_model(data, run_dir, ...)         # autoencoder + dynamics + classifier
    pred  = interpolate_stvcr(model, target_time, ...)    # AnnData of predicted cells
    # or all three at once:
    pred  = run_stvcr(support_adata, target_time, run_dir, ...)

Things this wrapper handles because the stVCR code needs them:
  * stVCR modifies the AnnData it is given (centres/rescales obsm['spatial'], adds X_ae ...), so we
    always work on a copy;
  * it hard-codes the names obs['time'] and obs['Annotation'];
  * the autoencoder reads adata.X.A, so X must be a scipy sparse matrix;
  * train_stvcr() mutates the config dict it receives (num_samples becomes a per-timepoint list),
    so a FRESH dict is built for every run;
  * torch >= 2.6 refuses to load the pickled models unless weights_only=False;
  * the interpolated AnnData stores colours as dict_values (not writable) and has no gene names.

Frame of the output: obsm['X_spatial_aligned'] is stVCR's normalised frame (each slice centred, divided
by adata.uns['spatial_scale_factor'], plus learned rigid transforms). obsm['spatial'] is that same
frame multiplied back by the scale factor, i.e. in raw coordinate UNITS, but centred/aligned.
"""
import functools
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

# the settings used in the stVCR ARTISTA tutorial notebook (03_ARTISTA_dynamics)
TUTORIAL_CONFIG = {
    "learning_rate": 1e-3, "learning_rate_rigid": 1e-4, "n_epochs": 2001, "num_samples": 1000,
    "lambda_match": 4e5, "alpha_exp": 0.01, "alpha_gro": 0.0002, "kappa_exp": 0.02, "kappa_gro": 0.1,
}

_patched = False


def _allow_pickled_torch_load():
    """torch>=2.6 defaults to weights_only=True, which cannot load stVCR's pickled model objects."""
    global _patched
    if not _patched:
        import torch
        torch.load = functools.partial(torch.load, weights_only=False)
        _patched = True


def prepare_stvcr_data(adata, time_key="time", spatial_key="spatial", label_key="Annotation", colors=None,
                       min_genes=100, min_cells=3, n_top_genes=2000, batch_key="Batch"):
    """
    Copy of `adata` ready for stVCR. Cell/gene filtering and the highly-variable-gene selection follow
    the tutorial; the thresholds are clamped for small gene panels (min_genes <= 10% of genes, and HVG
    selection is skipped when n_top_genes >= number of genes).
    """
    import scanpy as sc

    a = adata.copy()
    a.obs["time"] = np.asarray(a.obs[time_key], dtype=float)
    present = pd.Series(a.obs[label_key].astype(str).values)
    cats = [c for c in (colors if colors is not None else sorted(present.unique())) if c in set(present)]
    a.obs["Annotation"] = pd.Categorical(present.values, categories=cats)
    a.uns["Annotation_colors"] = [colors[c] for c in cats] if colors is not None else \
        [f"C{i}" for i in range(len(cats))]
    a.X = sp.csr_matrix(a.X).astype(np.float32)
    for k in list(a.obsm.keys()):
        if k != spatial_key:
            del a.obsm[k]                                   # e.g. an old X_ae would make stVCR skip its autoencoder
    a.obsm[spatial_key] = np.asarray(a.obsm[spatial_key], dtype=np.float32)
    if batch_key not in a.obs:
        a.obs[batch_key] = a.obs["time"].astype(str).values

    min_genes = int(min(min_genes, max(1, a.n_vars // 10)))
    sc.pp.filter_cells(a, min_genes=min_genes)
    sc.pp.filter_genes(a, min_cells=min_cells)
    if n_top_genes < a.n_vars:
        sc.pp.highly_variable_genes(a, n_top_genes=int(n_top_genes), batch_key=batch_key, subset=True)
    print(f"[stvcr] prepared {a.n_obs} cells x {a.n_vars} genes, timepoints "
          f"{np.unique(a.obs['time']).tolist()} (min_genes={min_genes}, min_cells={min_cells})")
    return a


def train_stvcr_model(adata, run_dir, spatial_key="spatial", n_epochs=2001, num_samples=1000,
                      z_dims=10, ae_epochs=1000, use_initial_alignment=True, seed=19491001,
                      config_overrides=None):
    """
    Autoencoder -> stVCR dynamics (with rigid alignment) -> time-dependent cell-type classifier.
    `adata` comes from prepare_stvcr_data and is modified in place. Model files go to run_dir.
    """
    import stvcr

    _allow_pickled_torch_load()
    os.makedirs(run_dir, exist_ok=True)
    paths = {k: os.path.join(run_dir, f"{k}.pth") for k in ("ae", "model", "rigid", "classifier")}

    stvcr.pp.pp_stvcr(adata=adata, use_pp_with_scanpy=False, ae_model_save_path=paths["ae"], z_dims=z_dims,
                      n_epochs=ae_epochs, seed=seed, use_initial_alignment=use_initial_alignment,
                      spatial_key=spatial_key, time_key="time")

    config = dict(TUTORIAL_CONFIG)                         # fresh dict every run (train_stvcr mutates it)
    config.update(config_overrides or {})
    config["n_epochs"], config["num_samples"] = int(n_epochs), int(num_samples)
    stvcr.tr.train_stvcr(adata=adata, model_path=paths["model"], rigid_transformation_path=paths["rigid"],
                         config=config)

    label_map = stvcr.dsm.create_spatiotemporal_classifier(
        adata, st_classifier_save_path=paths["classifier"], annotation_key="Annotation")
    return {"adata": adata, "paths": paths, "label_map": label_map,
            "scale": float(adata.uns["spatial_scale_factor"]),
            "params": dict(n_epochs=n_epochs, num_samples=num_samples, z_dims=z_dims, ae_epochs=ae_epochs,
                           use_initial_alignment=use_initial_alignment, seed=seed, **(config_overrides or {}))}


def interpolate_stvcr(model, target_time, delta_t=0.1, time_unit="day", native_png=None):
    """
    Evolve the nearest earlier observed slice forward to `target_time` with the learned dynamics
    (Euler steps of size delta_t, random division/death from the learned growth rate).
    """
    import matplotlib.pyplot as plt
    import stvcr

    _allow_pickled_torch_load()
    adata, paths = model["adata"], model["paths"]
    pred = stvcr.dsm.interpolate(
        adata, model_path=paths["model"], st_classifier_path=paths["classifier"],
        label_to_cell_type_map=model["label_map"], interpolation_time=float(target_time),
        ae_model_path=paths["ae"], delta_t=delta_t, time_unit=time_unit,
        save_path=native_png or os.path.join(os.path.dirname(paths["model"]), "stvcr_native.png"),
        show_or_save="save", cell_type_colors_key="Annotation_colors")
    plt.close("all")

    pred.var_names = list(adata.var_names)                 # the decoder returns genes in the training order
    pred.uns["Annotation_colors"] = list(pred.uns["Annotation_colors"])
    pred.obsm["spatial"] = (np.asarray(pred.obsm["X_spatial_aligned"], dtype=np.float64)
                            * model["scale"]).astype(np.float32)
    return pred


def run_stvcr(support_adata, target_time, run_dir, time_key="time", spatial_key="spatial",
              label_key="Annotation", colors=None, min_genes=100, min_cells=3, n_top_genes=2000,
              n_epochs=2001, num_samples=1000, z_dims=10, ae_epochs=1000, use_initial_alignment=True,
              delta_t=0.1, time_unit="day", seed=19491001):
    """
    Prepare, train and interpolate in one call. The label column comes back named `label_key`.

    A target AFTER the last support timepoint (forward extrapolation) needs no special handling: stVCR
    evolves the latest observed slice forward until the target time.
    Backward extrapolation is not supported.
    """
    data = prepare_stvcr_data(support_adata, time_key, spatial_key, label_key, colors,
                              min_genes, min_cells, n_top_genes)
    model = train_stvcr_model(data, run_dir, spatial_key, n_epochs, num_samples, z_dims, ae_epochs,
                              use_initial_alignment, seed)
    pred = interpolate_stvcr(model, target_time, delta_t, time_unit)
    if label_key != "Annotation":
        pred.obs[label_key] = pred.obs.pop("Annotation")
        pred.uns[f"{label_key}_colors"] = pred.uns.pop("Annotation_colors")
    pred.uns["stvcr_params_json"] = str(model["params"])
    return pred