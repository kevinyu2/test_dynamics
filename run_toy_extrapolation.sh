#!/bin/bash
# run_toy_extrapolation.sh -- forward extrapolation of timepoints 3 and 4 on the toy dataset, on SLURM.
#
#   bash run_toy_extrapolation.sh                   # full settings, results in out_toy_extrap/
#   bash run_toy_extrapolation.sh out_x --quick     # first argument = output folder, the rest goes to run_pipeline.py
#
# What it runs (times 1..4, predicting from EARLIER timepoints only; there is no backward extrapolation):
#   target 3  <- trained on 1, 2        moscot, stvcr_all   (stvcr_two would be identical to stvcr_all, so it is skipped)
#   target 4  <- trained on 2, 3        moscot, stvcr_two
#             <- trained on 1, 2, 3     stvcr_all
# = 5 array jobs plus a merge job. Set the interpreters / partition at the top of submit_slurm.sh as usual.
# When the jobs have finished (the merge job writes out/predictions):
#   python score.py --pred-dir out_toy_extrap/predictions --original toy_all.h5ad --out out_toy_extrap/scores
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"

OUT=${1:-out_toy_extrap}
[ $# -gt 0 ] && shift
bash submit_slurm.sh --input toy_all.h5ad --out "$OUT" --mode extrapolate --targets 3 4 "$@"

echo
echo "after the jobs finish:"
echo "  python score.py --pred-dir $OUT/predictions --original toy_all.h5ad --out $OUT/scores"
