"""python -m pytest -q test_prepare_artista.py   -- prepare_artista.py on a small fake file shaped like Regeneration.h5ad"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

import prepare_artista

BATCHES = {2: "Injury_2DPI_rep1_X", 5: "Injury_5DPI_rep1_X", 10: "Injury_10DPI_rep1_X", 15: "Injury_15DPI_rep4_X",
           20: "Injury_20DPI_rep2_X", 30: "Injury_30DPI_rep2_X", "control": "Injury_control_X"}


def fake(n=50, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for k, name in BATCHES.items():
        a = ad.AnnData(X=rng.random((n, 4)).astype(np.float32))
        a.obs["Batch"] = name
        a.obs["Annotation"] = "A"
        a.obsm["spatial"] = np.column_stack([rng.uniform(3500, 6000, n), rng.uniform(9000, 12000, n)])  # keep region of every rule
        blocks.append(a)
    full = ad.concat(blocks)
    full.obs_names = ["cell"] * full.n_obs                              # duplicated names, like the real file
    return full


def test_time_column_batches_and_unique_names():
    out, removed = prepare_artista.prepare(fake())
    assert removed == 0
    assert sorted(out.obs["time"].unique()) == [2.0, 5.0, 10.0, 15.0, 20.0]       # 30 DPI and control dropped
    assert out.n_obs == 250 and out.obs_names.is_unique
    assert set(out.obs["Batch"]) == {BATCHES[d] for d in (2, 5, 10, 15, 20)}       # Batch kept for HVG selection


def test_only_the_tutorial_outliers_are_removed():
    full = fake()
    xy = np.asarray(full.obsm["spatial"]).copy()
    is10, is15, is20, is2 = (full.obs["Batch"].values == BATCHES[d] for d in (10, 15, 20, 2))
    xy[np.flatnonzero(is10)[0]] = [30000, 12000]      # x > 28000 on 10 DPI        -> removed
    xy[np.flatnonzero(is15)[0]] = [7000, 12000]       # x > 6050 on 15 DPI         -> removed
    xy[np.flatnonzero(is15)[1]] = [2000, 8200]        # y<8500 & x<3000 on 15 DPI  -> removed
    xy[np.flatnonzero(is20)[0]] = [12000, 6000]       # y < 7000 on 20 DPI         -> removed
    xy[np.flatnonzero(is2)[0]] = [30000, 1000]        # same coordinates on 2 DPI have no rule -> kept
    full.obsm["spatial"] = xy
    out, removed = prepare_artista.prepare(full)
    assert removed == 4 and out.n_obs == 246
    kept, _ = prepare_artista.prepare(full, drop_outliers=False)
    assert kept.n_obs == 250


def test_other_days_and_missing_day():
    out, _ = prepare_artista.prepare(fake(), days=[5, 30])
    assert sorted(out.obs["time"].unique()) == [5.0, 30.0]
    with pytest.raises(ValueError, match="60"):
        prepare_artista.prepare(fake(), days=[5, 60])
    bad = fake()
    del bad.obsm["spatial"]
    with pytest.raises(KeyError):
        prepare_artista.prepare(bad)


def test_cli_roundtrip_feeds_the_pipeline_planner(tmp_path):
    import common
    src, dst = tmp_path / "in.h5ad", tmp_path / "out.h5ad"
    fake().write_h5ad(src)
    prepare_artista.main(["--input", str(src), "--out", str(dst)])
    t = ad.read_h5ad(dst).obs["time"].values
    runs = common.plan_runs(np.unique(t))
    runs = [r for r in runs if "skip" not in r]
    assert {r["mode"] for r in runs} == {"interpolate", "extrapolate"} and len(runs) == 17
