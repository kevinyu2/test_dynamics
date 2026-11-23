"""python -m pytest -q test_ablation.py -- the 'which timepoints are seen' experiment"""
import os
import stat
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import ablation
import common

HERE = os.path.dirname(os.path.abspath(__file__))


def test_variants_remove_every_combination_of_the_middle_timepoints():
    vs = ablation.variants([2, 5, 10, 15, 20], 10)
    assert [(v[0], v[1], v[2]) for v in vs] == [
        ("drop_none", "all", [2, 5, 10, 15, 20]), ("drop_5", "-5", [2, 10, 15, 20]),
        ("drop_15", "-15", [2, 5, 10, 20]), ("drop_5_15", "-5 -15", [2, 10, 20])]
    # explicit candidates; first / last / target can never be removed
    assert [v[0] for v in ablation.variants([1, 2, 3, 4, 5, 6], 3, [2, 5])] == ["drop_none", "drop_2", "drop_5", "drop_2_5"]
    for bad in ([1], [6], [3]):
        with pytest.raises(ValueError, match="cannot remove"):
            ablation.variants([1, 2, 3, 4, 5, 6], 3, bad)
    with pytest.raises(ValueError, match="first or last"):
        ablation.variants([1, 2, 3], 3)
    with pytest.raises(ValueError, match="nothing to remove"):
        ablation.variants([1, 2, 3], 2)
    with pytest.raises(ValueError, match="not among"):
        ablation.variants([1, 2, 3], 7)
    assert ablation.variants([1, 2.5, 4, 5], 4, [2.5])[1][0] == "drop_2p5"


def test_the_pipeline_plan_for_each_variant_sees_the_right_timepoints():
    plans = {}
    for label, _, kept in ablation.variants([2, 5, 10, 15, 20], 10):
        runs = {r["model"]: r for r in common.plan_runs(kept, [10], mode="interpolate")}
        plans[label] = {m: (r["support"], r["flank"], "skip" in r) for m, r in runs.items()}
    assert plans["drop_none"]["moscot"][0] == [5, 15] and plans["drop_none"]["stvcr_all"][0] == [2, 5, 15, 20]
    assert plans["drop_5"]["moscot"][:2] == ([2, 15], [2, 15]) and plans["drop_5"]["stvcr_all"][0] == [2, 15, 20]
    assert plans["drop_15"]["moscot"][0] == [5, 20]
    assert plans["drop_5_15"]["stvcr_all"][0] == [2, 20] and plans["drop_5_15"]["stvcr_two"][2]   # identical -> skipped
    assert not plans["drop_5"]["stvcr_two"][2]


def fake_input(path):
    rng = np.random.default_rng(0)
    parts = []
    for t in (2, 5, 10, 15, 20):
        a = ad.AnnData(rng.normal(size=(30, 4)).astype(np.float32))
        a.obs["time"] = float(t)
        a.obsm["spatial"] = rng.normal(size=(30, 2))
        parts.append(a)
    ad.concat(parts).write_h5ad(path)


def test_expand_rewrites_arguments_per_variant(tmp_path, capsys):
    inp = tmp_path / "d.h5ad"
    fake_input(inp)
    ablation.main(["expand", "--input", str(inp), "--out", str(tmp_path / "o"), "--targets", "5", "--mode", "both",
                   "--drop-ablation", "10", "--quick", "--stvcr-epochs", "50", "--models", "moscot", "stvcr_all"])
    out = capsys.readouterr().out.strip().splitlines()
    assert [l.split("\t")[0] for l in out] == ["drop_none", "drop_5", "drop_15", "drop_5_15"]
    line = out[1].split("\t")[1]
    assert f"--out {tmp_path}/o/drop_5 " in line and "--times 2 10 15 20 --targets 10 --mode interpolate" in line
    assert "--quick --stvcr-epochs 50 --models moscot stvcr_all" in line          # everything else passed through
    assert "--drop-ablation" not in line and "--targets 5" not in line


def test_submit_script_submits_every_variant(tmp_path):
    inp = tmp_path / "d.h5ad"
    fake_input(inp)
    fake = tmp_path / "bin"
    fake.mkdir()
    sb = fake / "sbatch"
    sb.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$SBATCH_LOG"\nprintf -- "---\\n" >> "$SBATCH_LOG"\necho 4242\n')
    sb.chmod(sb.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", SBATCH_LOG=str(tmp_path / "log"))
    r = subprocess.run(["bash", os.path.join(HERE, "submit_slurm.sh"), "--input", str(inp), "--out", str(tmp_path / "o"),
                        "--drop-ablation", "10", "--models", "moscot", "stvcr_all", "stvcr_two"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for d in ("drop_none", "drop_5", "drop_15", "drop_5_15"):
        jobs = (tmp_path / "o" / d / "slurm" / "jobs.txt").read_text().strip().splitlines()
        assert all("--targets 10" in j and "--mode interpolate" in j for j in jobs)
    counts = {d: len((tmp_path / "o" / d / "slurm" / "jobs.txt").read_text().strip().splitlines())
              for d in ("drop_none", "drop_5", "drop_15", "drop_5_15")}
    assert counts == {"drop_none": 3, "drop_5": 3, "drop_15": 3, "drop_5_15": 2}   # stvcr_two skipped when == stvcr_all
    assert len((tmp_path / "log").read_text().split("---\n")) - 1 == 8              # an array + a merge job per variant
    times = {d: (tmp_path / "o" / d / "slurm" / "jobs.txt").read_text().splitlines()[0].split("--times ")[1].split(" --")[0]
             for d in counts}
    assert times == {"drop_none": "2 5 10 15 20", "drop_5": "2 10 15 20", "drop_15": "2 5 10 20", "drop_5_15": "2 10 20"}


def test_score_collects_all_variants_into_one_comparison(tmp_path, monkeypatch):
    inp = tmp_path / "d.h5ad"
    fake_input(inp)
    full = ad.read_h5ad(inp)
    rng = np.random.default_rng(1)
    colors = {"a": "#000000"}
    for label, flank in (("drop_none", [5, 15]), ("drop_5", [2, 15])):
        d = tmp_path / "o" / label / "predictions"
        d.mkdir(parents=True)
        for m in ("moscot", "stvcr_all"):
            p = ad.AnnData(rng.normal(size=(25, 4)).astype(np.float32))
            p.var_names = full.var_names
            p.obsm["spatial"] = rng.normal(size=(25, 2))
            p = common.finalize_prediction(p, model=m, target=10.0, support=flank, flank=flank, label_key="Annotation",
                                           colors=colors, params={})
            p.write_h5ad(d / f"{m}__t10.h5ad")
    long = ablation.main(["score", "--out", str(tmp_path / "o"), "--original", str(inp), "--no-gw", "--no-frit",
                          "--n-max", "20", "--n-repeats", "1"])
    assert set(long.variant) == {"drop_none", "drop_5"}
    got = long[long.metric == "w2_expr"]
    assert {(r.variant, r.model) for r in got.itertuples()} >= {("drop_5", "moscot"), ("drop_none", "stvcr_all"),
                                                                ("drop_5", "observed_t2"), ("drop_none", "observed_t5")}
    plots = tmp_path / "o" / "ablation_scores" / "plots"
    assert (plots / "w2_expr.png").stat().st_size > 1000 and (tmp_path / "o" / "ablation_scores" / "ablation_long.csv").exists()
    again = ablation.main(["score", "--out", str(tmp_path / "o"), "--plot-only"])     # replot without rescoring
    assert len(again) == len(long)
