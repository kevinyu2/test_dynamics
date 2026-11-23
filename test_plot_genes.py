"""python -m pytest -q test_plot_genes.py -- gene-expression panels, on a small fake dataset and fake predictions"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

import common
import plot_genes

GENES = ["Sox2", "Gfap", "Actb"]


def make_original(path, n=80, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        a = ad.AnnData(X=rng.gamma(2.0, 1.0, (n, 3)).astype(np.float32) + t * 0.1)
        a.var_names = GENES
        a.obs_names = [f"t{t:g}_{i}" for i in range(n)]
        a.obs["time"] = t
        a.obsm["spatial"] = rng.normal(size=(n, 2)) * 100 + t * 50
        blocks.append(a)
    ad.concat(blocks).write_h5ad(path)


def make_pred(pred_dir, model, target, direction=None, genes=GENES, n=60, seed=1):
    rng = np.random.default_rng(seed)
    p = ad.AnnData(X=(rng.normal(size=(n, len(genes))) + 1).astype(np.float32))   # includes negative values
    p.var_names = genes
    p.obsm["spatial"] = rng.normal(size=(n, 2)) * 100
    p.write_h5ad(pred_dir / f"{common.pred_name(model, target, direction)}.h5ad")


@pytest.fixture()
def setup(tmp_path):
    orig = tmp_path / "orig.h5ad"
    make_original(orig)
    pd_ = tmp_path / "pred"
    pd_.mkdir()
    return str(orig), pd_, tmp_path


def args(orig, pred_dir, out, target, neighbors, genes, *extra):
    return plot_genes.parse_args(["--original", orig, "--pred-dir", str(pred_dir), "--out", str(out), "--target",
                                  str(target), "--neighbors", *map(str, neighbors), "--genes", *genes, *extra])


@pytest.fixture()
def spy(monkeypatch):
    figs = []
    real = plot_genes.build_figure

    def wrapper(*a, **k):
        fig = real(*a, **k)
        figs.append(fig)
        return fig

    monkeypatch.setattr(plot_genes, "build_figure", wrapper)
    return figs


def test_interpolation_four_panels_one_file_per_method_and_gene(setup, spy, capsys):
    orig, pred_dir, tmp = setup
    make_pred(pred_dir, "moscot", 3.0)
    make_pred(pred_dir, "stvcr_all", 3.0)                           # stvcr_two missing -> skipped with a message
    written = plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["Sox2", "Gfap"]))
    names = sorted(p.split("/")[-1] for p in written)
    assert names == ["gene_Gfap__moscot__t3.png", "gene_Gfap__stvcr_all__t3.png",
                     "gene_Sox2__moscot__t3.png", "gene_Sox2__stvcr_all__t3.png"]
    assert all((tmp / "out" / n).stat().st_size > 1000 for n in names)
    assert "stvcr_two: no stvcr_two__t3.h5ad" in capsys.readouterr().out
    assert all(len(f.axes) == 4 + 1 for f in spy)                   # left, true, predicted, right + colour bar


def test_extrapolation_three_panels_reads_fwd_files(setup, spy):
    orig, pred_dir, tmp = setup
    make_pred(pred_dir, "moscot", 4.0, "forward")
    make_pred(pred_dir, "moscot", 4.0)                              # an interpolation file must NOT be used
    written = plot_genes.run(args(orig, pred_dir, tmp / "out", 4, [3], ["Sox2"], "--models", "moscot"))
    assert [p.split("/")[-1] for p in written] == ["gene_Sox2__moscot__fwd_t4.png"]
    assert len(spy[0].axes) == 3 + 1


def test_all_panels_share_one_colour_scale_taken_from_real_slices(setup, spy):
    orig, pred_dir, tmp = setup
    make_pred(pred_dir, "moscot", 3.0)
    plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["Sox2"], "--models", "moscot"))
    fig = spy[0]
    clims = {tuple(np.round(ax.collections[0].get_clim(), 9)) for ax in fig.axes[:4]}
    assert len(clims) == 1
    real = ad.read_h5ad(orig)
    pool = np.concatenate([real[np.isclose(real.obs["time"], t)][:, "Sox2"].X.ravel() for t in (2.0, 3.0, 4.0)])
    lo, hi = clims.pop()
    assert lo == pytest.approx(np.percentile(pool, 0.5)) and hi == pytest.approx(np.percentile(pool, 99.5))


def test_points_sorted_low_to_high_and_downsampled(setup, spy):
    orig, pred_dir, tmp = setup
    make_pred(pred_dir, "moscot", 3.0)
    plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["Sox2"], "--models", "moscot", "--max-points", "30"))
    for ax in spy[0].axes[:4]:
        v = np.asarray(ax.collections[0].get_array())
        assert len(v) == 30 and np.all(np.diff(v) >= 0)


def test_gene_resolution_missing_genes_and_models_without_the_gene(setup, capsys):
    orig, pred_dir, tmp = setup
    make_pred(pred_dir, "moscot", 3.0)
    make_pred(pred_dir, "stvcr_all", 3.0, genes=["Actb"])           # like stVCR: only some genes were predicted
    written = plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["sox2"]))   # case-insensitive
    assert [p.split("/")[-1] for p in written] == ["gene_Sox2__moscot__t3.png"]
    assert "'Sox2' is not among its 1 predicted genes" in capsys.readouterr().out
    with pytest.raises(ValueError, match="similar names.*Sox2"):
        plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["Sox"]))
    with pytest.raises(ValueError, match="not found"):
        plot_genes.run(args(orig, pred_dir, tmp / "out", 3, [2, 4], ["Nope"]))


def test_time_validation(setup):
    orig, pred_dir, tmp = setup
    for target, nb in ((3, [3, 4]), (3, [2, 2.5]), (3, [4]), (3, [4, 5]), (3, [])):
        with pytest.raises((ValueError, SystemExit)):
            plot_genes.run(args(orig, pred_dir, tmp / "o", target, nb, ["Sox2"]))
    with pytest.raises(ValueError, match="no cells"):
        plot_genes.run(args(orig, pred_dir, tmp / "o", 3, [2, 9], ["Sox2"]))
    with pytest.raises(ValueError, match="backward"):
        plot_genes.check_times(1.0, [2.0])


def test_constant_gene_does_not_crash():
    lo, hi = plot_genes.color_range([np.zeros(10), np.zeros(5)])
    assert hi > lo


def test_artista_style_names_symbols_search_and_safe_file_names(tmp_path, capsys):
    names = ["SOX2 | AMEX60DD001640", "SOX3[nr]|SOX2[hs] | AMEX60DD037395", "GFAP | AMEX1", "GFAP | AMEX2"]
    assert plot_genes.resolve_gene("SOX2", names) == "SOX2 | AMEX60DD001640"       # not the alias SOX3|SOX2 entry
    assert plot_genes.resolve_gene("sox2", names) == "SOX2 | AMEX60DD001640"
    assert plot_genes.resolve_gene("GFAP | AMEX2", names) == "GFAP | AMEX2"
    assert plot_genes.resolve_gene("OTHER", names) is None
    with pytest.raises(ValueError, match="several genes.*AMEX1.*AMEX2"):
        plot_genes.resolve_gene("GFAP", names)
    assert plot_genes.file_safe("SOX2 | AMEX60DD001640") == "SOX2_AMEX60DD001640"

    orig = tmp_path / "o.h5ad"
    make_original(orig)
    a = ad.read_h5ad(orig)
    a.var_names = ["SOX2 | AMEX1", "GFAP | AMEX2", "ACTB | AMEX3"]
    a.write_h5ad(orig)
    pdir = tmp_path / "p"
    pdir.mkdir()
    make_pred(pdir, "moscot", 3.0, genes=["SOX2 | AMEX1", "GFAP | AMEX2", "ACTB | AMEX3"])
    make_pred(pdir, "stvcr_all", 3.0, genes=["ACTB | AMEX3"])
    written = plot_genes.run(args(str(orig), pdir, tmp_path / "out", 3, [2, 4], ["SOX2"], "--models", "moscot"))
    assert [w.split("/")[-1] for w in written] == ["gene_SOX2_AMEX1__moscot__t3.png"]
    plot_genes.main(["--original", str(orig), "--pred-dir", str(pdir), "--target", "3", "--search", "actb"])
    out = capsys.readouterr().out
    assert "1 genes contain 'actb'" in out and "ACTB | AMEX3   [moscot, stvcr_all]" in out
    plot_genes.main(["--original", str(orig), "--pred-dir", str(pdir), "--target", "3", "--search", "sox"])
    assert "SOX2 | AMEX1   [moscot]" in capsys.readouterr().out
