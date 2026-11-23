#!/usr/bin/env python
"""
ablation.py -- "which other timepoints does the model get to see?" for ONE interpolation target.

Holds out one timepoint (say 10) and repeats the usual interpolation (all three methods) while removing some of
the OTHER timepoints from the data. With timepoints 2 5 10 15 20 and target 10 the removable ones are 5 and 15
(everything except the target and the first/last timepoint, so there is always something on each side), and every
combination is run:

    all        (nothing removed: the normal run)      moscot / stvcr_two:  5 -> 15      stvcr_all: 2 5 15 20
    -5                                                moscot / stvcr_two:  2 -> 15      stvcr_all: 2 15 20
    -15                                               moscot / stvcr_two:  5 -> 20      stvcr_all: 2 5 20
    -5 -15                                            moscot / stvcr_two:  2 -> 20      stvcr_all: 2 20

Submit everything (one SLURM submission per variant, each in its own folder OUT/drop_<...>/):

    bash submit_slurm.sh --input data.h5ad --out out_drop --drop-ablation 10 [any other flag, e.g. --quick]
    bash submit_slurm.sh ... --drop-ablation 10 --drop-candidates 5 15      # choose which timepoints may be removed

When everything finished, score all variants and compare them:

    sbatch --wrap "python ablation.py score --out out_drop --original data.h5ad"
    python ablation.py score --out out_drop --original data.h5ad --plot-only        # redraw only

writes OUT/drop_<...>/scores (as score.py would) and OUT/ablation_scores/{ablation_long.csv, plots/}: one bar plot
per metric, groups = variants, bars = models (the grey bar is the better of the two observed neighbours of that variant).
Everything after `score` that this script does not know is passed on to score.py (e.g. --no-gw).
"""
import argparse
import itertools
import os
import shlex
import sys

import numpy as np
import pandas as pd

import common

MAX_VARIANTS = 16


def variants(times, target, candidates=None):
    """-> [(label, display label, kept times)] for every subset of `candidates` that is removed, fewest removed first."""
    times = sorted(float(x) for x in times)
    hit = [t for t in times if np.isclose(t, float(target))]
    if not hit:
        raise ValueError(f"target {target:g} is not among the timepoints {times}")
    target = hit[0]
    if target in (times[0], times[-1]):
        raise ValueError(f"target {target:g} is the first or last timepoint: it cannot be interpolated")
    if candidates is None:
        cands = [t for t in times if t not in (times[0], times[-1], target)]
    else:
        cands = []
        for c in candidates:
            m = [t for t in times if np.isclose(t, float(c))]
            if not m:
                raise ValueError(f"--drop-candidates {c:g} is not among the timepoints {times}")
            if m[0] in (times[0], times[-1]) or m[0] == target:
                raise ValueError(f"cannot remove {c:g}: only timepoints strictly between the first and last, "
                                 f"other than the target, may be removed")
            cands.append(m[0])
        cands = sorted(set(cands))
    if not cands:
        raise ValueError(f"nothing to remove: no timepoint other than {target:g} lies between {times[0]:g} and {times[-1]:g}")
    out = []
    for k in range(len(cands) + 1):
        for drop in itertools.combinations(cands, k):
            label = "drop_" + "_".join(common.time_tag(t) for t in drop) if drop else "drop_none"
            shown = " ".join(f"-{t:g}" for t in drop) if drop else "all"
            out.append((label, shown, [t for t in times if t not in drop]))
    return out


def _parser():
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--input")
    p.add_argument("--out")
    p.add_argument("--time-key", default="time")
    p.add_argument("--times", type=float, nargs="*", default=None)
    p.add_argument("--targets", type=float, nargs="*", default=None)
    p.add_argument("--mode", default=None)
    p.add_argument("--drop-ablation", type=float, default=None)
    p.add_argument("--drop-candidates", type=float, nargs="*", default=None)
    return p


def expand(argv):
    """The submit_slurm.sh arguments -> one line per variant: '<label>\\t<rewritten arguments>'."""
    a, extra = _parser().parse_known_args(argv)
    if a.drop_ablation is None or not a.input or not a.out:
        raise SystemExit("--drop-ablation <target time>, --input and --out are all required")
    if a.targets:
        print("[ablation] ignoring --targets: the target is the --drop-ablation value", file=sys.stderr)
    if a.mode not in (None, "interpolate"):
        print(f"[ablation] ignoring --mode {a.mode}: this experiment is interpolation only", file=sys.stderr)
    import anndata as ad
    t = np.asarray(ad.read_h5ad(a.input, backed="r").obs[a.time_key], dtype=float)
    times = np.unique(t)
    if a.times:
        times = np.array([x for x in times if any(np.isclose(x, w) for w in a.times)])
    vs = variants(times, a.drop_ablation, a.drop_candidates)
    if len(vs) > MAX_VARIANTS:
        raise SystemExit(f"{len(vs)} variants (x 3 methods each) is a lot; pass --drop-candidates with fewer timepoints "
                         f"(more than {MAX_VARIANTS} variants is refused)")
    print(f"[ablation] {len(vs)} variants for target {a.drop_ablation:g}: " + ", ".join(v[1] for v in vs), file=sys.stderr)
    for label, _, kept in vs:
        args = ["--input", a.input, "--out", os.path.join(a.out, label), "--time-key", a.time_key,
                "--times", *[f"{x:g}" for x in kept], "--targets", f"{a.drop_ablation:g}", "--mode", "interpolate", *extra]
        print(label + "\t" + " ".join(shlex.quote(x) for x in args))


def score(out, original, extra, plot_only=False):
    import score as scoring
    dirs = sorted(d for d in os.listdir(out) if d.startswith("drop_") and os.path.isdir(os.path.join(out, d, "predictions")))
    if not dirs:
        raise SystemExit(f"no drop_*/predictions folders in {out}")
    longs = []
    for d in dirs:
        sdir = os.path.join(out, d, "scores")
        if not plot_only:
            print(f"\n[ablation] scoring {d}")
            scoring.main(["--pred-dir", os.path.join(out, d, "predictions"), "--original", original, "--out", sdir,
                          "--no-plots", *extra])
        csv = os.path.join(sdir, "scores_long.csv")
        if not os.path.exists(csv):
            print(f"[ablation] {d}: no scores yet, left out")
            continue
        longs.append(pd.read_csv(csv).assign(variant=d))
    if not longs:
        raise SystemExit("nothing was scored")
    long = pd.concat(longs, ignore_index=True)
    # the variant label takes the place of the "task" so score.py's bar plots group by variant
    shown = {}
    for d in dirs:
        dropped = d[len("drop_"):]
        shown[d] = "all" if dropped == "none" else " ".join("-" + x.replace("p", ".") for x in dropped.split("_"))
    order = sorted(set(long.variant), key=lambda v: (len(v.split("_")) if v != "drop_none" else 0, v))
    for v in order:
        label = shown[v]
        scoring.TASK_ORDER.append(label)
        scoring.GROUP_LABEL[label] = label.replace("{", "{{").replace("}", "}}")
    odir = os.path.join(out, "ablation_scores")
    os.makedirs(odir, exist_ok=True)
    long.to_csv(os.path.join(odir, "ablation_long.csv"), index=False)
    plot_long = long.assign(task=long.variant.map(shown))
    paths = scoring.plot_scores(plot_long, os.path.join(odir, "plots"))
    pd.set_option("display.width", 200)
    for m in [m for m in long.metric.unique() if m != "frit_angle_deg"]:
        t = plot_long[plot_long.metric == m].pivot_table(index="model", columns="task", values="mean", sort=False)
        print(f"\n== {m} (rows: model, columns: removed timepoints; lower is better)")
        print(t.round(4).to_string())
    print(f"\n[ablation] wrote {odir}/ablation_long.csv and {len(paths)} plots in {odir}/plots/")
    return long


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("expand", "score"):
        raise SystemExit(__doc__)
    cmd, rest = argv[0], argv[1:]
    if cmd == "expand":
        return expand(rest)
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--out", required=True, help="the --out given to submit_slurm.sh")
    p.add_argument("--original", help="the full dataset (not needed with --plot-only)")
    p.add_argument("--plot-only", action="store_true")
    a, extra = p.parse_known_args(rest)
    if not a.plot_only and not a.original:
        p.error("--original is required")
    return score(a.out, a.original, extra, a.plot_only)


if __name__ == "__main__":
    main()
