#!/usr/bin/env python
"""
prepare_mosta.py -- turn MOSTA's Mouse_embryo_all_stage.h5ad (520,815 cells, E9.5-E16.5) into a pipeline input.

What the original has: obs['timepoint'] as text ("E9.5"), obs['annotation'] (45 cell types), obsm['spatial'], a CSR
matrix X (23,761 genes) and, probably, raw counts in layers['count']. What the pipeline needs (see common.py): a numeric
obs['time'], obs['Annotation'], log-normalised X, and a size moscot can solve. This script

  * writes time = the number in "E9.5" (9.5 ... 16.5, days of embryonic development),
  * keeps --n-per-time cells per timepoint (default 10000, random; timepoints with fewer keep all). moscot's plan is a
    dense (cells x cells) matrix per pair of timepoints, so ~10k is the practical size; raise it together with the job's
    memory. Cell-type proportions are preserved, very rare types can vanish,
  * keeps the --n-genes most variable genes (default 5000; 0 = all). moscot predicts every gene it is given and writes
    them densely, so all 23,761 would make huge prediction files. stVCR selects its own 2000 variable genes anyway,
  * sets Annotation (+ Annotation_colors if the file has annotation_colors), Batch = the timepoint label,
  * reads the 22 GB file in row blocks with h5py: only the chosen cells are kept in memory.

X: with --source auto (default) the script looks at X. If it holds non-integer values it is taken as already
log-normalised and used as it is; if it holds integers it is treated as counts and normalised (10,000 per cell, log1p).
--source count uses layers['count'] and normalises it, --source X forces X as it is.

    python prepare_mosta.py --input Mouse_embryo_all_stage.h5ad --out mosta_10k.h5ad
    python prepare_mosta.py --input ... --out mosta_small.h5ad --n-per-time 3000 --times 9.5 10.5 11.5 12.5 13.5
It prints, per timepoint, the cell counts and the extent of the coordinates: check that each looks like ONE tissue section.
Needs ~30 GB of memory and several minutes on a compute node (the file is read once), e.g.
    sbatch --mem=48G --time=02:00:00 --cpus-per-task=4 --wrap "python prepare_mosta.py --input ... --out ..."
"""
import argparse
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp


def parse_time(label):
    m = re.search(r"(\d+(?:\.\d+)?)", str(label))
    if not m:
        raise ValueError(f"cannot read a number from timepoint '{label}'")
    return float(m.group(1))


def read_rows(h5file, path, rows, n_vars, chunk=25000):
    """Rows `rows` (sorted, unique) of the matrix at `path` (CSR group or dense dataset) -> scipy CSR, float32."""
    import h5py
    node = h5file[path]
    rows = np.asarray(rows)
    parts = []
    if isinstance(node, h5py.Group):
        enc = node.attrs.get("encoding-type", "")
        if enc != "csr_matrix":
            raise ValueError(f"{path} is stored as '{enc}'; only csr_matrix or a dense array is supported here")
        indptr = node["indptr"][:]
        n_rows = len(indptr) - 1
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            lo = np.searchsorted(rows, start)
            hi = np.searchsorted(rows, stop)
            if lo == hi:
                continue
            a, b = indptr[start], indptr[stop]
            block = sp.csr_matrix((node["data"][a:b], node["indices"][a:b], indptr[start:stop + 1] - a),
                                  shape=(stop - start, n_vars))
            parts.append(block[rows[lo:hi] - start])
    else:
        for start in range(0, node.shape[0], chunk):
            stop = min(start + chunk, node.shape[0])
            lo, hi = np.searchsorted(rows, start), np.searchsorted(rows, stop)
            if lo != hi:
                parts.append(sp.csr_matrix(node[start:stop][rows[lo:hi] - start]))
    return sp.vstack(parts, format="csr").astype(np.float32) if parts else sp.csr_matrix((0, n_vars), dtype=np.float32)


def looks_like_counts(h5file, path, n_vars, n_probe=3000):
    m = read_rows(h5file, path, np.arange(n_probe), n_vars, chunk=n_probe)
    return m.nnz > 0 and bool(np.all(m.data == np.round(m.data)))


def normalise(X, target=1e4):
    X = sp.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(1)).ravel()
    tot[tot == 0] = 1.0
    X = sp.diags((target / tot).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    return sp.csr_matrix(X)


def prepare(path, n_per_time=10000, n_genes=5000, source="auto", times=None, seed=0, chunk=25000,
            time_col="timepoint", label_col="annotation", spatial_key="spatial", min_cells=3):
    import anndata as ad
    import h5py

    backed = ad.read_h5ad(path, backed="r")
    obs, spatial = backed.obs.copy(), np.asarray(backed.obsm[spatial_key], dtype=np.float32)
    var_names = list(backed.var_names)
    colors = backed.uns.get(f"{label_col}_colors")
    backed.file.close()
    n_vars = len(var_names)
    for col in (time_col, label_col):
        if col not in obs:
            raise KeyError(f"obs['{col}'] not found; columns are {list(obs.columns)}")

    tlabel = obs[time_col].astype(str).values
    tnum = np.array([parse_time(x) for x in tlabel])
    keep_times = np.unique(tnum) if not times else np.array([t for t in np.unique(tnum) if any(np.isclose(t, w) for w in times)])
    if len(keep_times) < 3:
        raise ValueError(f"need at least 3 timepoints, got {keep_times.tolist()}")

    rng = np.random.default_rng(seed)
    chosen = []
    print(f"{'time':>6} {'cells':>8} {'kept':>7}   x range / y range (raw units)")
    for t in keep_times:
        idx = np.flatnonzero(np.isclose(tnum, t))
        pick = np.sort(rng.choice(idx, n_per_time, replace=False)) if n_per_time and len(idx) > n_per_time else idx
        chosen.append(pick)
        xy = spatial[pick]
        print(f"{t:6g} {len(idx):8d} {len(pick):7d}   x {xy[:, 0].min():.0f}..{xy[:, 0].max():.0f}   "
              f"y {xy[:, 1].min():.0f}..{xy[:, 1].max():.0f}")
    rows = np.sort(np.concatenate(chosen))

    with h5py.File(path, "r") as f:
        if source == "auto":
            if looks_like_counts(f, "X", n_vars):
                src, norm = ("layers/count", True) if "layers" in f and "count" in f["layers"] else ("X", True)
            else:
                src, norm = "X", False
        elif source == "count":
            src, norm = "layers/count", True
        else:
            src, norm = "X", False
        print(f"matrix: {src} ({'normalised here: 1e4 per cell + log1p' if norm else 'used as is, assumed log-normalised'})")
        X = read_rows(f, src, rows, n_vars, chunk)
    if norm:
        X = normalise(X)

    a = ad.AnnData(X=X)
    a.var_names = var_names
    if not a.var_names.is_unique:
        print("warning: duplicate gene names, made unique")
        a.var_names_make_unique()
    a.obs_names = [f"{tlabel[r]}_{r}" for r in rows]
    a.obs["time"] = tnum[rows]
    a.obs["Batch"] = pd.Categorical(tlabel[rows])
    lab = obs[label_col].iloc[rows]
    cats = list(lab.cat.categories) if hasattr(lab, "cat") else sorted(lab.astype(str).unique())
    present = [c for c in cats if (lab.astype(str) == str(c)).any()]
    a.obs["Annotation"] = pd.Categorical(lab.astype(str).values, categories=[str(c) for c in present])
    if colors is not None and len(colors) == len(cats):
        cmap = dict(zip([str(c) for c in cats], [str(c) for c in colors]))
        a.uns["Annotation_colors"] = [cmap[str(c)] for c in present]
    a.obsm["spatial"] = spatial[rows]

    keep = np.asarray(a.X.getnnz(axis=0)) >= min_cells
    a = a[:, keep].copy()
    if n_genes and n_genes < a.n_vars:
        import scanpy as sc
        sc.pp.highly_variable_genes(a, n_top_genes=int(n_genes), flavor="seurat", batch_key="Batch", subset=True)
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    print(f"result: {a.n_obs} cells x {a.n_vars} genes, timepoints {np.unique(a.obs['time']).tolist()}, "
          f"{a.obs['Annotation'].nunique()} cell types")
    return a


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-per-time", type=int, default=10000, help="cells kept per timepoint (0 = all)")
    p.add_argument("--n-genes", type=int, default=5000, help="most variable genes kept (0 = all)")
    p.add_argument("--source", choices=["auto", "X", "count"], default="auto")
    p.add_argument("--times", type=float, nargs="*", default=None, help="only these timepoints, e.g. 9.5 10.5 11.5")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out = prepare(a.input, a.n_per_time, a.n_genes, a.source, a.times, a.seed)
    out.write_h5ad(a.out)
    print(f"wrote {a.out}")
    print(f"Next:  bash submit_slurm.sh --input {a.out} --out mosta_run --moscot-epsilon 0.03 --moscot-tau 0.97 ...")
    return out


if __name__ == "__main__":
    main()
