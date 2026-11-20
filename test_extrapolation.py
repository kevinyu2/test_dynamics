"""
test_extrapolation.py -- tests for the extrapolation mode (planning, moscot extrapolation, pipeline, scoring).

    python -m pytest -q test_extrapolation.py

moscot and stVCR themselves are not needed: the moscot interpolation step is tested with a hand-made coupling,
and the full pipeline (plan -> predict -> write -> manifest -> score -> plots) runs with the model call stubbed out.
"""
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import common
import run_parallel
import run_pipeline
import score
from call_moscot import interpolate_moscot


# ------------------------------------------------------------------------------------------------ planning
def by_key(runs):
    return {(r["model"], r["direction"], r["target"]): r for r in runs}


def test_names_mark_forward_extrapolation_and_keep_interpolation_unchanged():
    assert common.pred_name("moscot", 2.0) == "moscot__t2"
    assert common.pred_name("moscot", 2.0, None) == "moscot__t2"
    assert common.pred_name("stvcr_all", 4.0, "forward") == "stvcr_all__fwd_t4"
    assert common.pred_name("moscot", 2.5, "forward") == "moscot__fwd_t2p5"


def test_interpolation_plan_is_unchanged():
    runs = common.plan_runs([1, 2, 3, 4], mode="interpolate")
    assert {r["target"] for r in runs} == {2.0, 3.0}
    assert all(r["mode"] == "interpolate" and r["direction"] is None for r in runs)
    r = by_key(runs)[("moscot", None, 2.0)]
    assert r["support"] == [1.0, 3.0] and r["flank"] == [1.0, 3.0]


def test_extrapolation_targets_and_supports():
    runs = by_key(common.plan_runs([1, 2, 3, 4, 5], mode="extrapolate"))
    assert sorted({k[2] for k in runs}) == [3.0, 4.0, 5.0]               # needs two earlier timepoints
    assert {k[1] for k in runs} == {"forward"}
    # forward target 4: only timepoints before it; moscot / stvcr_two use the two nearest, stvcr_all all of them
    assert runs[("moscot", "forward", 4.0)]["support"] == [2.0, 3.0]
    assert runs[("stvcr_two", "forward", 4.0)]["support"] == [2.0, 3.0]
    assert runs[("stvcr_all", "forward", 4.0)]["support"] == [1.0, 2.0, 3.0]
    assert runs[("moscot", "forward", 4.0)]["flank"] == [3.0]                 # baseline = nearest slice only
    # nothing from the far side of the target is ever used
    for (m, d, t), r in runs.items():
        assert all(s < t for s in r["support"])
        assert r["mode"] == "extrapolate" and r["direction"] == "forward"


def test_stvcr_two_skipped_when_identical_to_stvcr_all():
    runs = by_key(common.plan_runs([1, 2, 3, 4], mode="extrapolate"))
    assert "skip" in runs[("stvcr_two", "forward", 3.0)]                       # support [1,2] == all earlier
    assert "skip" not in runs[("stvcr_two", "forward", 4.0)]                   # [2,3] vs [1,2,3]


def test_target_selection_and_no_backward_option():
    fwd = common.plan_runs([1, 2, 3, 4], mode="extrapolate")
    assert {r["direction"] for r in fwd} == {"forward"} and {r["target"] for r in fwd} == {3.0, 4.0}
    only4 = common.plan_runs([1, 2, 3, 4], targets=[4], mode="extrapolate")
    assert {(r["direction"], r["target"]) for r in only4} == {("forward", 4.0)}
    for first_two in (1, 2):                                                    # no two earlier timepoints
        with pytest.raises(ValueError):
            common.plan_runs([1, 2, 3, 4], targets=[first_two], mode="extrapolate")
    with pytest.raises(TypeError):
        common.plan_runs([1, 2, 3, 4], mode="extrapolate", direction="backward")
    with pytest.raises(ValueError):
        common.plan_runs([1, 2, 3, 4], targets=[2.5], mode="extrapolate")
    with pytest.raises(ValueError):
        common.plan_runs([1, 2], mode="extrapolate")
    with pytest.raises(ValueError):
        common.plan_runs([1, 2, 3], mode="sideways")


# ------------------------------------------------------------------------------------------------ moscot
def make_model(P, shift, n_src, n_tgt, t_src=1.0, t_tgt=2.0, seed=0, tgt_label="A"):
    """src has random cells; every target cell sits at its (first) parent's position + shift."""
    rng = np.random.default_rng(seed)
    src = ad.AnnData(X=rng.random((n_src, 3)).astype(np.float32))
    src.obsm["spatial"] = rng.random((n_src, 2))
    src.obsm["X_pca"] = rng.random((n_src, 2))
    src.obs["Annotation"] = ["A"] * n_src
    owner = np.argmax(np.asarray(P) > 0, axis=0)                   # which source cell each target cell comes from
    tgt = ad.AnnData(X=(np.asarray(src.X)[owner] + 1.0).astype(np.float32))
    tgt.obsm["spatial"] = src.obsm["spatial"][owner] + shift
    tgt.obsm["X_pca"] = src.obsm["X_pca"][owner] + 0.5
    tgt.obs["Annotation"] = [tgt_label] * n_tgt
    for a in (src, tgt):
        a.var_names = ["g1", "g2", "g3"]
    return {"P": np.asarray(P), "src": src, "tgt": tgt, "t_src": t_src, "t_tgt": t_tgt,
            "spatial_key": "spatial", "label_key": "Annotation"}


SHIFT = np.array([2.0, -1.0])


@pytest.mark.parametrize("target,lam", [(3.0, 2.0), (2.5, 1.5), (1.5, 0.5)])
def test_identity_coupling_gives_exact_linear_continuation(target, lam):
    n = 40
    m = make_model(np.eye(n) / n, SHIFT, n, n)
    pred = interpolate_moscot(m, target, allow_extrapolation=True, seed=1)
    assert pred.n_obs == n                                          # equal counts stay equal
    idx = pred.obs["parent_idx"].values
    src_xy = m["src"].obsm["spatial"]
    # new cell = (1-lam) * early + lam * late; with partner == same index this is src + lam * SHIFT
    np.testing.assert_allclose(pred.obsm["spatial"], src_xy[idx] + lam * SHIFT, atol=1e-5)
    np.testing.assert_allclose(np.asarray(pred.X), np.asarray(m["src"].X)[idx] + lam * 1.0, atol=1e-5)
    np.testing.assert_allclose(pred.obsm["X_pca"], m["src"].obsm["X_pca"][idx] + lam * 0.5, atol=1e-5)


@pytest.mark.parametrize("target,expected", [(3.0, 16), (1.5, 6)])
def test_cell_count_follows_exponential_growth_law(target, expected):
    # 4 source cells -> 8 target cells (each source cell sends equal mass to two target cells)
    P = np.zeros((4, 8))
    for i in range(4):
        P[i, 2 * i] = P[i, 2 * i + 1] = 1 / 8
    m = make_model(P, SHIFT, 4, 8)
    pred = interpolate_moscot(m, target, allow_extrapolation=True)
    assert pred.n_obs == expected                                   # 4^(1-lam) * 8^lam: 16 / round(5.66)=6


def test_strict_by_default_and_rejects_observed_times():
    m = make_model(np.eye(10) / 10, SHIFT, 10, 10)
    with pytest.raises(ValueError):
        interpolate_moscot(m, 3.0)                                  # beyond the data without allow_extrapolation
    for before in (0.0, -1.0):
        for allow in (False, True):
            with pytest.raises(ValueError, match="backward"):
                interpolate_moscot(m, before, allow_extrapolation=allow)   # backward is never allowed
    for obs_t in (1.0, 2.0):
        with pytest.raises(ValueError):
            interpolate_moscot(m, obs_t, allow_extrapolation=True)  # coincides with an observed timepoint
    assert interpolate_moscot(m, 1.5).n_obs == 10                   # plain interpolation still works


def test_no_nans_when_some_source_cells_have_zero_growth():
    P = np.eye(10) / 10
    P[0, 0] = 0.0                                                   # source cell 0 sends no mass anywhere
    m = make_model(np.eye(10), SHIFT, 10, 10)
    m["P"] = P
    pred = interpolate_moscot(m, 3.0, allow_extrapolation=True)
    assert np.isfinite(pred.obsm["spatial"]).all() and np.isfinite(np.asarray(pred.X)).all()


@pytest.mark.parametrize("target,expected", [(1.25, "A"), (1.5, "A"), (1.75, "B"), (3.0, "B")])
def test_labels_inherit_from_the_nearer_endpoint(target, expected):
    # earlier slice is all A, later slice all B. lam=0.5 is a tie and goes to the parent (earlier cell).
    # forward extrapolation: the nearest observed slice is the later one, so it gets that one's label.
    n = 30
    m = make_model(np.eye(n) / n, SHIFT, n, n, tgt_label="B")
    pred = interpolate_moscot(m, target, allow_extrapolation=True)
    assert set(pred.obs["Annotation"]) == {expected}
    assert set(pred.obs["parent_label"]) | set(pred.obs["partner_label"]) == {"A", "B"}   # both kept


def test_no_label_or_direction_flags_left_on_the_command_line():
    for flag in ("--moscot-labels", "--k-label", "--direction"):
        with pytest.raises(SystemExit):
            run_pipeline.parse_args(["--input", "x.h5ad", "--out", "o", flag, "forward"])


# ------------------------------------------------------------------------------------------------ pipeline + score (model call stubbed)
def make_dataset(path, n=60, seed=0):
    rng = np.random.default_rng(seed)
    blocks = []
    for t in (1.0, 2.0, 3.0, 4.0, 5.0):
        lab = rng.choice(["A", "B"], n)
        a = ad.AnnData(X=(rng.normal(size=(n, 5)) + t).astype(np.float32))
        a.var_names = [f"g{i}" for i in range(5)]
        a.obs_names = [f"t{t:g}_{i}" for i in range(n)]
        a.obs["time"] = t
        a.obs["Annotation"] = pd.Categorical(lab, categories=["A", "B"])
        a.obsm["spatial"] = rng.normal(size=(n, 2)) * 5 + np.where(lab == "A", 0, 12)[:, None] + t
        blocks.append(a)
    full = ad.concat(blocks)
    full.obs["Annotation"] = pd.Categorical(full.obs["Annotation"].astype(str), categories=["A", "B"])
    full.write_h5ad(path)
    return full


@pytest.fixture()
def stubbed(monkeypatch, tmp_path):
    """run_one returns a copy of the nearest observed slice, i.e. exactly the naive baseline."""
    data = tmp_path / "toy.h5ad"
    full = make_dataset(str(data))
    calls = []

    def fake_run_one(run, adata, args, colors, out):
        calls.append(run)
        t = np.asarray(adata.obs[args.time_key], dtype=float)
        pred = adata[np.isclose(t, run["flank"][0])].copy()
        pred.obs_names = [f"fake_{i}" for i in range(pred.n_obs)]
        return pred, {"stub": True}

    monkeypatch.setattr(run_pipeline, "run_one", fake_run_one)
    return str(data), full, calls, tmp_path


def run_extrap_pipeline(data, out, *extra):
    return run_pipeline.main(["--input", data, "--out", str(out), "--mode", "extrapolate", "--models", "moscot",
                              "stvcr_all", *extra])


def test_pipeline_extrapolation_writes_named_files_and_metadata(stubbed):
    data, full, calls, tmp = stubbed
    out = tmp / "out"
    rows = run_extrap_pipeline(data, out)
    names = sorted(os.listdir(out / "predictions"))
    # times 1..5 -> forward targets 3,4,5, for 2 models
    assert len(names) == 6 and "moscot__fwd_t5.h5ad" in names and not any("bwd" in n for n in names)
    assert all(r["status"] == "ok" for r in rows)
    assert {(r["mode"], r["direction"]) for r in rows} == {("extrapolate", "forward")}
    assert len(os.listdir(out / "images")) == 6
    p = ad.read_h5ad(out / "predictions" / "stvcr_all__fwd_t4.h5ad")
    assert str(p.uns["interp_mode"]) == "extrapolate" and str(p.uns["interp_direction"]) == "forward"
    assert list(p.uns["interp_support_times"]) == [1.0, 2.0, 3.0] and list(p.uns["interp_flank_times"]) == [3.0]
    assert float(p.uns["interp_target_time"]) == 4.0 and (p.obs["time"] == 4.0).all()
    # the model call only ever saw timepoints before its target
    for r in calls:
        assert all(s < r["target"] for s in r["support"])
    cfg = json.load(open(out / "config.json"))
    assert cfg["args"]["mode"] == "extrapolate"


def test_pipeline_targets_flag(stubbed):
    data, full, calls, tmp = stubbed
    run_extrap_pipeline(data, tmp / "o1", "--targets", "5")
    assert sorted(os.listdir(tmp / "o1" / "predictions")) == ["moscot__fwd_t5.h5ad", "stvcr_all__fwd_t5.h5ad"]


def test_interpolation_mode_still_default(stubbed):
    data, full, calls, tmp = stubbed
    run_pipeline.main(["--input", data, "--out", str(tmp / "oi"), "--models", "moscot", "--targets", "3",
                       "--mode", "interpolate"])
    assert os.listdir(tmp / "oi" / "predictions") == ["moscot__t3.h5ad"]
    p = ad.read_h5ad(tmp / "oi" / "predictions" / "moscot__t3.h5ad")
    assert str(p.uns["interp_mode"]) == "interpolate" and str(p.uns["interp_direction"]) == ""


def test_score_extrapolation_tasks_baseline_and_plots(stubbed):
    data, full, calls, tmp = stubbed
    out = tmp / "out"
    run_extrap_pipeline(data, out)
    sc = tmp / "scores"
    score.main(["--pred-dir", str(out / "predictions"), "--original", data, "--out", str(sc),
                "--n-max", "100", "--n-repeats", "1"])
    long = pd.read_csv(sc / "scores_long.csv")
    assert set(long.task) == {"extrap_fwd"}
    # exactly one observed baseline per (task, target): the nearest slice
    obs = long[long.model.str.startswith("observed")]
    assert obs.groupby(["task", "target_time"]).model.nunique().eq(1).all()
    assert set(obs[(obs.task == "extrap_fwd") & (obs.target_time == 4.0)].model) == {"observed_t3"}
    assert set(obs[(obs.task == "extrap_fwd") & (obs.target_time == 5.0)].model) == {"observed_t4"}
    # the stub returns the nearest slice, so every metric must tie exactly with the observed baseline
    for (task, tgt, metric), g in long.groupby(["task", "target_time", "metric"]):
        base = g[g.model.str.startswith("observed")]["mean"].iloc[0]
        for v in g[~g.model.str.startswith("observed")]["mean"]:
            assert v == pytest.approx(base, abs=1e-6), (task, tgt, metric)
    plots = os.listdir(sc / "plots")
    assert "all_metrics.png" in plots and "w2_expr.png" in plots
    wide = pd.read_csv(sc / "scores_wide.csv")
    assert "task" in wide.columns


def test_plot_groups_and_labels():
    rows = []
    for task, tgt in [("extrap_fwd", 4.0), ("interp", 2.0)]:
        for model in ("moscot", "stvcr_all", "observed_t9"):
            rows.append(dict(model=model, task=task, target_time=tgt, metric="w2_expr", mean=1.0, std=0.1))
    groups, series, label = score._bar_data(pd.DataFrame(rows), "w2_expr")
    assert groups == [("interp", 2.0), ("extrap_fwd", 4.0)]      # fixed task order
    assert list(series) == ["moscot", "stvcr_all", "observed"]
    assert label == "nearest observed slice"                                            # one baseline per group
    rows += [dict(model="observed_t8", task="interp", target_time=2.0, metric="w2_expr", mean=0.5, std=0.1)]
    groups, series, label = score._bar_data(pd.DataFrame(rows), "w2_expr")
    assert series["observed"][0][0] == 0.5                                              # better of 2 for interp
    assert label == "observed neighbour (best available)"


def test_old_scores_without_task_column_still_plot(tmp_path):
    rows = [dict(model=m, target_time=2.0, metric="w2_expr", mean=1.0, std=0.1) for m in ("moscot", "observed_t1")]
    pd.DataFrame(rows).to_csv(tmp_path / "scores_long.csv", index=False)
    score.main(["--plot-only", "--out", str(tmp_path)])
    assert (tmp_path / "plots" / "w2_expr.png").exists()


# ------------------------------------------------------------------------------------------------ parallel launcher
def test_parallel_dry_run_lists_one_job_per_run(stubbed, capsys):
    data, full, calls, tmp = stubbed
    run_parallel.main(["--input", data, "--out", str(tmp / "par"), "--mode", "extrapolate", "--dry-run"])
    lines = [l for l in capsys.readouterr().out.splitlines() if "run_pipeline.py" in l]
    # 5 timepoints: forward targets 3,4,5; moscot + stvcr_all each; stvcr_two skipped only where identical (t3)
    assert len(lines) == 3 * 2 + 2
    one = next(l for l in lines if "stvcr_all__fwd_t4" in l)
    assert "--models stvcr_all" in one and "--targets 4" in one and "--mode extrapolate" in one
    assert all("--direction" not in l for l in lines)
    # interpolation jobs carry no direction and keep their old names
    run_parallel.main(["--input", data, "--out", str(tmp / "par2"), "--mode", "interpolate", "--dry-run"])
    inter = [l for l in capsys.readouterr().out.splitlines() if "run_pipeline.py" in l]
    assert inter and all("--direction" not in l and "--mode interpolate" in l for l in inter)
    assert any("moscot__t2 " in l or "/moscot__t2 " in l for l in inter)


def test_toy_extrapolation_of_3_and_4_plans_five_jobs(capsys):
    here = os.path.dirname(os.path.abspath(__file__))
    run_parallel.main(["--input", os.path.join(here, "toy_all.h5ad"), "--out", "/tmp/toy_dry", "--mode", "extrapolate",
                       "--targets", "3", "4", "--dry-run"])
    lines = [l for l in capsys.readouterr().out.splitlines() if "run_pipeline.py" in l]
    got = sorted(l.split("--models ")[1].split()[0] + "@" + l.split("--targets ")[1].split()[0] for l in lines)
    assert got == ["moscot@3", "moscot@4", "stvcr_all@3", "stvcr_all@4", "stvcr_two@4"]


# ------------------------------------------------------------------------------------------------ mode both (default)
def test_default_mode_is_both_and_plans_interpolation_then_extrapolation():
    runs = common.plan_runs([1, 2, 3, 4, 5])
    modes = [r["mode"] for r in runs]
    assert modes == sorted(modes, key=lambda m: m != "interpolate")                  # all interpolation runs first
    interp = {(r["model"], r["target"]) for r in runs if r["mode"] == "interpolate"}
    extrap = {(r["model"], r["target"]) for r in runs if r["mode"] == "extrapolate"}
    assert {t for _, t in interp} == {2.0, 3.0, 4.0} and {t for _, t in extrap} == {3.0, 4.0, 5.0}
    assert runs == common.plan_runs([1, 2, 3, 4, 5], mode="both")
    assert [r for r in runs if r["mode"] == "interpolate"] == common.plan_runs([1, 2, 3, 4, 5], mode="interpolate")
    assert [r for r in runs if r["mode"] == "extrapolate"] == common.plan_runs([1, 2, 3, 4, 5], mode="extrapolate")
    names = [common.pred_name(r["model"], r["target"], r["direction"]) for r in runs]
    assert len(names) == len(set(names))                                              # file names never clash


def test_both_targets_are_used_for_every_mode_they_are_valid_for():
    runs = common.plan_runs([1, 2, 3, 4], targets=[3, 4])
    got = {(r["mode"], r["target"]) for r in runs}
    assert got == {("interpolate", 3.0), ("extrapolate", 3.0), ("extrapolate", 4.0)}   # 4 is the last: no interpolation
    assert {(r["mode"], r["target"]) for r in common.plan_runs([1, 2, 3, 4], targets=[2])} == {("interpolate", 2.0)}
    for bad in ([1], [2.5]):                                                           # first timepoint / not a timepoint
        with pytest.raises(ValueError):
            common.plan_runs([1, 2, 3, 4], targets=bad)
    with pytest.raises(ValueError):
        common.plan_runs([1, 2])


def test_pipeline_defaults_to_both_and_scores_have_both_tasks(stubbed):
    data, full, calls, tmp = stubbed
    out = tmp / "ob"
    run_pipeline.main(["--input", data, "--out", str(out), "--models", "moscot"])
    names = sorted(os.listdir(out / "predictions"))
    assert names == ["moscot__fwd_t3.h5ad", "moscot__fwd_t4.h5ad", "moscot__fwd_t5.h5ad",
                     "moscot__t2.h5ad", "moscot__t3.h5ad", "moscot__t4.h5ad"]
    sc = tmp / "sb"
    score.main(["--pred-dir", str(out / "predictions"), "--original", data, "--out", str(sc),
                "--n-max", "100", "--n-repeats", "1"])
    long = pd.read_csv(sc / "scores_long.csv")
    assert set(long.task) == {"interp", "extrap_fwd"}
    assert {"interp_t3", "extrap_fwd_t3"} <= {f"{t}_t{g:g}" for t, g in zip(long.task, long.target_time)}


def test_parallel_default_mode_both_job_count(stubbed, capsys):
    data, full, calls, tmp = stubbed
    run_parallel.main(["--input", data, "--out", str(tmp / "pb"), "--dry-run"])
    lines = [l for l in capsys.readouterr().out.splitlines() if "run_pipeline.py" in l]
    # 5 timepoints: interpolation 3 targets x 3 models = 9; extrapolation 3 targets x 2 models + stvcr_two at 4, 5 = 8
    assert len(lines) == 9 + 8
    assert sum("--mode interpolate" in l for l in lines) == 9 and sum("--mode extrapolate" in l for l in lines) == 8


# ------------------------------------------------------------------------------------------------ plan checks / GPU handling
def test_check_plan_accepts_good_plans_and_names_the_problem():
    from call_moscot import check_plan
    P = np.eye(5) / 5
    P[0, 0] = 0.0                                                   # a source cell with no growth is fine
    assert check_plan(P, converged=True)["zero_rows"] == 1
    for bad, word in ((np.full((3, 3), np.nan), "NaN"), (np.full((3, 3), np.inf), "NaN/inf"), (np.zeros((3, 3)), "zero total mass")):
        with pytest.raises(RuntimeError, match=word):
            check_plan(bad)
    m = make_model(np.eye(10) / 10, SHIFT, 10, 10)
    m["P"] = np.full((10, 10), np.nan)
    with pytest.raises(RuntimeError, match="unusable"):                 # instead of "Probabilities contain NaN"
        interpolate_moscot(m, 3.0, allow_extrapolation=True)
    m["P"] = np.zeros((10, 10))
    with pytest.raises(RuntimeError, match="unusable"):
        interpolate_moscot(m, 1.5)


def sbatch_calls(tmp_path, env):
    import stat
    import subprocess
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    sb = fake / "sbatch"
    sb.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$SBATCH_LOG"\nprintf -- "---\\n" >> "$SBATCH_LOG"\necho 4242\n')
    sb.chmod(sb.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "sbatch.log"
    here = os.path.dirname(os.path.abspath(__file__))
    e = {k: v for k, v in os.environ.items() if k not in ("GRES", "CUDA_VISIBLE_DEVICES")}
    e.update(PATH=f"{fake}:{e['PATH']}", SBATCH_LOG=str(log), **env)
    r = subprocess.run(["bash", os.path.join(here, "submit_slurm.sh"), "--input", os.path.join(here, "toy_all.h5ad"),
                        "--out", str(tmp_path / "o"), "--models", "moscot", "--mode", "interpolate"],
                       env=e, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return log.read_text().split("---\n")


def test_submit_slurm_hides_gpus_unless_gres_is_set(tmp_path):
    array, merge = sbatch_calls(tmp_path, {})[:2]
    assert 'export CUDA_VISIBLE_DEVICES=""' in array and "--gres" not in array
    (tmp_path / "g").mkdir()
    array2 = sbatch_calls(tmp_path / "g", {"GRES": "gpu:1"})[0]
    assert "--gres=gpu:1" in array2 and "CUDA_VISIBLE_DEVICES" not in array2
