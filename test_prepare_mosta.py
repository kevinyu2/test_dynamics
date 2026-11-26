"""python -m pytest -q test_prepare_mosta.py -- prepare_mosta.py on a small file with the layout of Mouse_embryo_all_stage.h5ad"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import prepare_mosta

TIMES = ["E9.5", "E10.5", "E11.5", "E12.5"]


def make_file(path, integer_x=False, dense_layer=False, n_per=60, n_genes=40, seed=0):
    rng = np.random.default_rng(seed)
    n = n_per * len(TIMES)
    counts = rng.poisson(0.7, (n, n_genes)).astype(np.float32)
    counts[:, 0] = 0                                           # a gene expressed nowhere
    counts[:2, 1] = 1                                          # a gene in only 2 cells
    counts[2:, 1] = 0
    counts[counts.sum(1) == 0, 2] = 1
    lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4).astype(np.float32)
    a = ad.AnnData(X=sp.csr_matrix(counts if integer_x else lognorm))
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["timepoint"] = pd.Categorical(np.repeat(TIMES, n_per))
    a.obs["annotation"] = pd.Categorical(rng.choice(["Brain", "Heart", "Liver", "Lung"], n))
    a.obs["total_counts"] = counts.sum(1)
    a.obsm["spatial"] = rng.normal(size=(n, 2)).astype(np.float32) * 1000 + np.repeat(np.arange(4), n_per)[:, None] * 50
    a.uns["annotation_colors"] = ["#111111", "#222222", "#333333", "#444444"]
    a.layers["count"] = counts if dense_layer else sp.csr_matrix(counts)
    a.write_h5ad(path)
    return a, counts, lognorm


def test_subsamples_per_timepoint_numeric_time_labels_and_colours(tmp_path):
    src = tmp_path / "m.h5ad"
    full, _, lognorm = make_file(src)
    out = prepare_mosta.prepare(str(src), n_per_time=25, n_genes=0, chunk=37)       # chunk size that splits timepoints
    assert out.n_obs == 100 and sorted(np.unique(out.obs["time"])) == [9.5, 10.5, 11.5, 12.5]
    assert all((out.obs["time"] == t).sum() == 25 for t in (9.5, 10.5, 11.5, 12.5))
    assert out.obsm["spatial"].shape == (100, 2) and out.obsm["spatial"].dtype == np.float32
    assert list(out.obs["Batch"].unique().sort_values()) == sorted(TIMES) and out.obs_names.is_unique
    # the right cells, with the right values (X used as is: it holds non-integers)
    orig = out.obs_names.str.split("_").str[1].astype(int)
    assert np.allclose(out.X.toarray(), lognorm[orig][:, [g for g in range(40) if g in set(map(int, out.var_names.str[1:]))]])
    assert np.allclose(out.obsm["spatial"], full.obsm["spatial"][orig])
    ann = dict(zip(["Brain", "Heart", "Liver", "Lung"], ["#111111", "#222222", "#333333", "#444444"]))
    assert list(out.uns["Annotation_colors"]) == [ann[c] for c in out.obs["Annotation"].cat.categories]
    assert (out.obs["Annotation"].astype(str).values == full.obs["annotation"].astype(str).values[orig]).all()


def test_small_timepoints_are_kept_whole_and_times_can_be_restricted(tmp_path):
    src = tmp_path / "m.h5ad"
    make_file(src, n_per=30)
    assert prepare_mosta.prepare(str(src), n_per_time=500, n_genes=0).n_obs == 120
    out = prepare_mosta.prepare(str(src), n_per_time=0, n_genes=0, times=[9.5, 10.5, 11.5])
    assert out.n_obs == 90 and 12.5 not in set(out.obs["time"])
    with pytest.raises(ValueError, match="at least 3"):
        prepare_mosta.prepare(str(src), times=[9.5, 10.5])


def test_gene_filter_and_variable_gene_selection(tmp_path):
    src = tmp_path / "m.h5ad"
    make_file(src)
    out = prepare_mosta.prepare(str(src), n_per_time=60, n_genes=0)
    assert "g0" not in out.var_names and "g1" not in out.var_names             # expressed nowhere / in only 2 cells
    out = prepare_mosta.prepare(str(src), n_per_time=60, n_genes=15)
    assert out.n_vars == 15


def test_integer_x_is_normalised_and_count_source_works(tmp_path):
    src = tmp_path / "m.h5ad"
    _, counts, lognorm = make_file(src, integer_x=True)
    for source in ("auto", "count", "X"):
        out = prepare_mosta.prepare(str(src), n_per_time=0, n_genes=0, source=source)
        orig = out.obs_names.str.split("_").str[1].astype(int)
        genes = [int(g[1:]) for g in out.var_names]
        want = counts[orig][:, genes] if source == "X" else lognorm[orig][:, genes]
        assert np.allclose(out.X.toarray(), want, atol=1e-5), source


def test_dense_layer_and_cli(tmp_path, capsys):
    src = tmp_path / "m.h5ad"
    _, _, lognorm = make_file(src, integer_x=True, dense_layer=True)
    out_path = tmp_path / "o.h5ad"
    prepare_mosta.main(["--input", str(src), "--out", str(out_path), "--n-per-time", "20", "--n-genes", "0"])
    back = ad.read_h5ad(out_path)
    assert back.n_obs == 80 and {"time", "Annotation", "Batch"} <= set(back.obs.columns) and "spatial" in back.obsm
    text = capsys.readouterr().out
    assert "matrix: layers/count (normalised here" in text and "9.5" in text
