#!/usr/bin/env python
"""
prepare_artista.py -- turn the ARTISTA axolotl-regeneration file (Regeneration.h5ad) into an input for the pipeline.

The pipeline needs a numeric `time` column in obs; the original file only has a `Batch` column such as
"Injury_10DPI_rep1_SS200000147BL_B5". This script
  * keeps the injury batches whose days-post-injury (DPI) are in --days (default 2 5 10 15 20, the range used in the
    stVCR tutorial; the uninjured "control" batch has no DPI and is dropped),
  * writes `time` = DPI (days),
  * removes the 20 stray cells the stVCR tutorial removes (coordinate thresholds on the 10, 15 and 20 DPI slices;
    --keep-outliers to skip), and
  * makes the cell names unique (the original file has duplicates).
Gene filtering / highly-variable-gene selection is NOT done here: the pipeline does it per model, as in the tutorial.
`Batch`, `Annotation` and obsm['spatial'] are kept as they are.

    python prepare_artista.py --input Regeneration.h5ad --out artista_2_20dpi.h5ad
    python prepare_artista.py --input Regeneration.h5ad --out artista_all.h5ad --days 2 5 10 15 20 30 60
"""
import argparse
import re

import anndata as ad
import numpy as np

# the tutorial's outlier rules: (DPI) -> function(x, y) -> boolean mask of cells to drop
OUTLIERS = {
    10: lambda x, y: x > 28000,
    15: lambda x, y: (x > 6050) | (y < 8000) | ((y < 8500) & (x < 3000)),
    20: lambda x, y: (y < 7000) | (x > 16000),
}


def dpi_of(batch):
    m = re.search(r"Injury_(\d+)DPI", str(batch))
    return float(m.group(1)) if m else np.nan


def prepare(adata, days=(2, 5, 10, 15, 20), drop_outliers=True, batch_key="Batch", spatial_key="spatial"):
    if batch_key not in adata.obs or spatial_key not in adata.obsm:
        raise KeyError(f"need obs['{batch_key}'] and obsm['{spatial_key}']; found obs {list(adata.obs.columns)}, "
                       f"obsm {list(adata.obsm.keys())}")
    dpi = np.array([dpi_of(b) for b in adata.obs[batch_key]])
    keep = np.isin(dpi, [float(d) for d in days])
    missing = sorted(set(float(d) for d in days) - set(dpi[keep]))
    if missing:
        raise ValueError(f"no batch found for DPI {missing}; batches present: {sorted(set(adata.obs[batch_key].astype(str)))}")
    a = adata[keep].copy()
    a.obs["time"] = dpi[keep]
    removed = 0
    if drop_outliers:
        xy = np.asarray(a.obsm[spatial_key], dtype=float)
        drop = np.zeros(a.n_obs, dtype=bool)
        for d, rule in OUTLIERS.items():
            on = a.obs["time"].values == float(d)
            drop |= on & rule(xy[:, 0], xy[:, 1])
        removed = int(drop.sum())
        a = a[~drop].copy()
    a.obs_names_make_unique()
    a.obs[batch_key] = a.obs[batch_key].astype(str)
    return a, removed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--days", type=float, nargs="+", default=[2, 5, 10, 15, 20])
    p.add_argument("--keep-outliers", action="store_true", help="do not remove the tutorial's stray cells")
    a = p.parse_args(argv)
    full = ad.read_h5ad(a.input)
    out, removed = prepare(full, a.days, drop_outliers=not a.keep_outliers)
    print(f"kept {out.n_obs} of {full.n_obs} cells ({removed} stray cells removed), {out.n_vars} genes")
    print(out.obs["time"].value_counts().sort_index().to_string())
    out.write_h5ad(a.out)
    print("wrote", a.out)
    return out


if __name__ == "__main__":
    main()
