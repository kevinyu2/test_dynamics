#!/bin/bash
# submit_slurm.sh -- one SLURM array task per (model, target), then a merge job.
#
#   bash submit_slurm.sh --input data.h5ad --out out [--quick --models ... --times ... any run_pipeline.py flag]
#
# By default (--mode both) it runs interpolation AND forward extrapolation. To restrict:
#   bash submit_slurm.sh --input data.h5ad --out out_interp --mode interpolate          # interpolation only
#   bash submit_slurm.sh --input data.h5ad --out out_extrap --mode extrapolate          # extrapolation only
#
# Every array task writes only into out/jobs/<job name>/ (no shared files), so tasks can't clash.
# Afterwards: sbatch --wrap "python score.py --pred-dir out/predictions --original data.h5ad --out out/scores"
#
# Run it from a shell where an env with anndata/pandas/numpy is active (it only plans the jobs). If stVCR and
# moscot live in different conda envs, set STVCR_PYTHON and MOSCOT_PYTHON below: each job then runs with the
# interpreter of its own env, in parallel, no activation needed.
set -eo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ---- edit these for your cluster -------------------------------------------------------------
PARTITION=""            # e.g. "cpu"; empty = cluster default
CPUS=8                  # cores per job
MEM=32G                 # memory per job
TIME=24:00:00           # wall time per job (stvcr_all is the slow one)
MAX_CONCURRENT=20       # the %N throttle on the array
ACTIVATE=':'            # commands run at the start of every job; ':' = inherit the env you submit from
STVCR_PYTHON=""         # e.g. /path/to/miniconda3/envs/stvcr/bin/python   (empty = use the submitting env's python)
MOSCOT_PYTHON=""        # e.g. /path/to/miniconda3/envs/moscot/bin/python
# -----------------------------------------------------------------------------------------------

eval "$ACTIVATE"
OUT=$(python - "$@" <<'PY'
import sys; a = sys.argv[1:]; print(a[a.index("--out") + 1])
PY
)
OUT=$(readlink -f "$OUT"); mkdir -p "$OUT/slurm"
JOBS="$OUT/slurm/jobs.txt"
python run_parallel.py "$@" --dry-run | grep run_pipeline.py > "$JOBS"
# each line starts with the python that planned the jobs; swap in the right env's python per model
if [ -n "$MOSCOT_PYTHON" ]; then sed -E -i "s#^[^ ]+ (.*--models moscot .*)\$#$MOSCOT_PYTHON \1#" "$JOBS"; fi
if [ -n "$STVCR_PYTHON" ];  then sed -E -i "s#^[^ ]+ (.*--models stvcr_.*)\$#$STVCR_PYTHON \1#" "$JOBS"; fi
N=$(wc -l < "$JOBS"); echo "$N jobs -> $JOBS"
PART=${PARTITION:+--partition=$PARTITION}

ARR=$(sbatch --parsable $PART --array=1-$N%$MAX_CONCURRENT --cpus-per-task=$CPUS --mem=$MEM --time=$TIME \
  --job-name=interp --output="$OUT/slurm/%A_%a.log" --wrap "
$ACTIVATE
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=\$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=\$SLURM_CPUS_PER_TASK PYTHONNOUSERSITE=1
cd $PWD
eval \"\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $JOBS)\"
")
echo "array job $ARR"

sbatch --parsable $PART --dependency=afterany:$ARR --cpus-per-task=1 --mem=8G --time=00:30:00 \
  --job-name=interp_merge --output="$OUT/slurm/merge_%j.log" --wrap "
$ACTIVATE
cd $PWD
python run_parallel.py $* --merge-only"
echo "merge job submitted (runs after the array finishes, even if some tasks failed)"
