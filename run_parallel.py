#!/usr/bin/env python
"""
run_parallel.py -- run run_pipeline.py as many independent processes, one per (model, target).

Every job gets its own output folder, so nothing can clash (run_pipeline.py's manifest.csv,
config.json and errors.log are shared files inside one --out):

    <out>/jobs/<model>__t<time>/     what run_pipeline.py wrote for that single job (runs/, logs of its own)
    <out>/logs/<model>__t<time>.log  stdout+stderr of that job
    <out>/predictions, images        merged at the end -> feed <out>/predictions to score.py
    <out>/manifest.csv               merged manifest (same columns as run_pipeline.py's)

Usage (everything unknown to this script is passed straight to run_pipeline.py):

    python run_parallel.py --input toy_all.h5ad --out out --gpus 0 1 2 3 --quick
    python run_parallel.py --input data.h5ad --out out --workers 8            # CPU only
    python run_parallel.py --input data.h5ad --out out --gpus 0 --jobs-per-gpu 3 --models stvcr_all
    python run_parallel.py --input data.h5ad --out out_extrap --mode extrapolate

Re-running is safe: finished predictions are skipped (add --overwrite to redo them).
Then:  python score.py --pred-dir <out>/predictions --original <input> --out <out>/scores
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import common
from common import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--time-key", default="time")
    p.add_argument("--times", type=float, nargs="*", default=None)
    p.add_argument("--targets", type=float, nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--mode", default="interpolate", choices=["interpolate", "extrapolate"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gpus", nargs="*", default=None, help="GPU ids to use, e.g. --gpus 0 1 2 3 (default: no GPU pinning)")
    p.add_argument("--jobs-per-gpu", type=int, default=1, help="concurrent jobs on each listed GPU")
    p.add_argument("--workers", type=int, default=None, help="concurrent jobs when --gpus is not given (default: 4)")
    p.add_argument("--threads", type=int, default=None, help="CPU threads per job (default: cores / concurrent jobs)")
    p.add_argument("--dry-run", action="store_true", help="print the jobs and commands, run nothing")
    p.add_argument("--merge-only", action="store_true", help="run nothing; just merge finished jobs/ into predictions/ (used after SLURM arrays)")
    return p.parse_known_args(argv)


def plan_jobs(a):
    import anndata as ad
    t = np.asarray(ad.read_h5ad(a.input, backed="r").obs[a.time_key], dtype=float)
    times = np.unique(t)
    if a.times:
        times = np.array([x for x in times if any(np.isclose(x, w) for w in a.times)])
    runs = [r for r in common.plan_runs(times, a.targets, a.models, mode=a.mode)
            if "skip" not in r]
    order = {"stvcr_all": 0, "stvcr_two": 1, "moscot": 2}          # slowest first -> better load balance
    runs.sort(key=lambda r: (order[r["model"]], r["direction"] or "", r["target"]))
    return runs


def jobname(run):
    return common.pred_name(run["model"], run["target"], run["direction"])


def main(argv=None):
    a, passthrough = parse_args(argv)
    out = os.path.abspath(a.out)
    a.input = os.path.abspath(a.input)                    # jobs run with cwd = this folder
    runs = plan_jobs(a)
    for sub in ("jobs", "logs", "predictions", "images"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    if a.gpus:
        slots = [g for g in a.gpus for _ in range(a.jobs_per_gpu)]
    else:
        slots = [None] * (a.workers or 4)
    slots = slots[: max(1, len(runs))]
    threads = a.threads or max(1, (os.cpu_count() or 1) // len(slots))
    free = queue.Queue()
    for s in slots:
        free.put(s)
    print(f"[parallel] {len(runs)} jobs, {len(slots)} at a time, {threads} CPU threads each, gpus={a.gpus or 'none pinned'}")

    def command(run, name):
        cmd = [sys.executable, os.path.join(HERE, "run_pipeline.py"), "--input", a.input,
               "--out", os.path.join(out, "jobs", name), "--time-key", a.time_key,
               "--models", run["model"], "--targets", f"{run['target']:g}", "--mode", run["mode"]]
        if a.times:
            cmd += ["--times"] + [f"{x:g}" for x in a.times]
        if a.overwrite:
            cmd.append("--overwrite")
        return cmd + passthrough

    def work(run):
        name = jobname(run)
        cmd = command(run, name)
        if os.path.exists(os.path.join(out, "predictions", f"{name}.h5ad")) and not a.overwrite:
            return name, "exists", 0.0
        slot = free.get()
        try:
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                       OPENBLAS_NUM_THREADS=str(threads), NUMBA_NUM_THREADS=str(threads),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",        # jax (moscot) would grab ~75% of the GPU per process
                       PYTHONUNBUFFERED="1")
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)
            print(f"[parallel] start  {name}" + (f" on gpu {slot}" if slot is not None else ""), flush=True)
            t0 = time.time()
            with open(os.path.join(out, "logs", f"{name}.log"), "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            dt = time.time() - t0
            ok = rc == 0 and os.path.exists(os.path.join(out, "jobs", name, "predictions", f"{name}.h5ad"))
            print(f"[parallel] {'done  ' if ok else 'FAILED'} {name} ({dt / 60:.1f} min)", flush=True)
            return name, "ok" if ok else "failed", dt
        finally:
            free.put(slot)

    if a.dry_run:
        for r in runs:
            print(" ".join(command(r, jobname(r))))
        return
    if a.merge_only:
        results = [(jobname(r), "ok" if os.path.exists(os.path.join(out, "jobs", jobname(r), "predictions",
                                                                    jobname(r) + ".h5ad")) else "failed", 0.0)
                   for r in runs]
    else:
        with ThreadPoolExecutor(max_workers=len(slots)) as ex:
            results = list(ex.map(work, runs))

    # ---- merge the per-job folders into one predictions/ images/ manifest.csv
    rows = []
    for name, status, _ in results:
        jd = os.path.join(out, "jobs", name)
        for sub, ext in (("predictions", ".h5ad"), ("images", ".png")):
            src = os.path.join(jd, sub, name + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out, sub, name + ext))
        mp = os.path.join(jd, "manifest.csv")
        if os.path.exists(mp):
            rows.append(pd.read_csv(mp))
    if rows:
        pd.concat(rows).drop_duplicates("file", keep="last").to_csv(os.path.join(out, "manifest.csv"), index=False)
    bad = [n for n, s, _ in results if s == "failed"]
    print(f"\n[parallel] {sum(s == 'ok' for _, s, _ in results)} ok, {sum(s == 'exists' for _, s, _ in results)} already there, "
          f"{len(bad)} failed{': ' + ', '.join(bad) if bad else ''}")
    if bad:
        print(f"  see {out}/logs/<job>.log and {out}/jobs/<job>/errors.log")
    print(f"Next:  python score.py --pred-dir {out}/predictions --original {a.input} --out {out}/scores")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
