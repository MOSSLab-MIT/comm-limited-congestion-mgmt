#!/usr/bin/env python3
import itertools, csv
from pathlib import Path
# ==========================================================

# Un-comment the baseline/method to run

# ========================== OURS ==========================
param_grid = {
    "torch_seed":       [0, 1, 2, 3, 4],
    "noise_mag":        [2.0, 2.4, 2.45, 2.5, 3.0, 3.5, 4.0],
    "offset":           [0, 60, 120, 180],
    "epochs":           [120],
    "radius":           [0.2],
    "lr":               [1e-3],
    "optimizer_type":   ["clipped_gd"],
    "lr_schedule":      ["plateau"],
    "patience":         [5],
    "lr_decay_step":    [20],
    "batch_size":       [2],
    "max_grad_norm":    [3e4]
}
pyfile    = "experiments/run_ours.py"             # script to run
tag       = "ours"                    # used in filenames/job name
conda_env = "INSERT CONDA ENV NAME"   # conda env

# ========================== OPT ===========================
# param_grid = {
#     "noise_mag":        [2.0, 2.4, 2.45, 2.5, 3.0, 3.5, 4.0],
#     "offset":           [0, 60, 120, 180],
#     "test_skew_mag":    [0, 10, 20, 30, 40],
#     "radius":           [0.2],
# }
# pyfile    = "experiments/run_opt.py"              # script to run
# tag       = "opt"                     # used in filenames/job name
# conda_env = "INSERT CONDA ENV NAME"   # conda env

# ========================== DEC ===========================
# param_grid = {
#     "noise_mag":        [2.0, 2.4, 2.45, 2.5, 3.0, 3.5, 4.0],
#     "offset":           [0, 60, 120, 180],
#     "test_skew_mag":    [0, 10, 20, 30, 40],
#     "radius":           [0.2],
# }
# pyfile    = "experiments/run_dec.py"              # script to run
# tag       = "dec"                     # used in filenames/job name
# conda_env = "INSERT CONDA ENV NAME"   # conda env

# ========================= PROXY ==========================
# param_grid = {
#     "torch_seed":       [0, 1, 2, 3, 4],
#     "noise_mag":        [2.0, 2.4, 2.45, 2.5, 3.0, 3.5, 4.0],
#     "offset":           [0, 60, 120, 180],
#     "epochs":           [120],
#     "radius":           [0.2],
#     "lr":               [1e-3],
#     "optimizer_type":   ["clipped_gd"],
#     "lr_schedule":      ["plateau"],
#     "patience":         [5],
#     "lr_decay_step":    [20],
#     "batch_size":       [2],
#     "max_grad_norm":    [3e4]
# }
# pyfile    = "experiments/run_proxy.py"             # script to run
# tag       = "proxy"                    # used in filenames/job name
# conda_env = "INSERT CONDA ENV NAME"    # conda env

# Slurm resources (tweak if needed)
partition = "mit_normal"
cpus      = "3"
mem       = "15G"
time_lim  = "12:00:00"
# ==========================================================

# Derived filenames
keys = list(param_grid.keys())
rows = list(itertools.product(*(param_grid[k] for k in keys)))
csv_name   = f"sweep_{tag}.csv"
slurm_name = f"run_sweep_{tag}.slurm"

# rerun_indices = [1, 2, 25, 29, 47]
# rows = [rows[i] for i in rerun_indices]

# Write sweep CSV
with open(csv_name, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(keys)
    for row in rows:
        w.writerow(row)
print(f"Wrote {csv_name} with {len(rows)} rows and header: {','.join(keys)}")

slurm_text = f"""#!/bin/bash
#SBATCH -J naza_sweep_{tag}
#SBATCH -p {partition}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time_lim}
#SBATCH --array=0-{len(rows)-1}
# Bootstrap logs
#SBATCH -o logs/bootstrap/bootstrap_%A_%a.out
#SBATCH -e logs/bootstrap/bootstrap_%A_%a.err

set -euo pipefail
mkdir -p logs

# >>> put all real logs under logs/${{SLURM_ARRAY_JOB_ID}}/ <<<
LOGDIR="logs/${{SLURM_ARRAY_JOB_ID}}"
mkdir -p "$LOGDIR"
# Redirect all further stdout/stderr to per-task files in LOGDIR
exec >  "${{LOGDIR}}/${{SLURM_JOB_NAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
exec 2> "${{LOGDIR}}/${{SLURM_JOB_NAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"
echo "Redirected logs to: $LOGDIR/${{SLURM_JOB_NAME}}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.{{out,err}}"

# --- Miniconda init + activate ---
module load miniforge/24.3.0-0
set +u
eval "$(conda shell.bash hook)"
conda activate RTE
set -u

# --- Gurobi init + activate, SPECIFIC TO MIT ENGAGING CLUSTER---
module load community-modules
module load gurobi/12.0.3

# --- make libs use the CPUs you asked for ---
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK}}
export MKL_DYNAMIC=FALSE
export OMP_PLACES=cores
export OMP_PROC_BIND=close

CSV="{csv_name}"
IDX=${{SLURM_ARRAY_TASK_ID}}

# Count usable rows (skip header & comment-only lines)
NUM_ROWS=$(awk -F, 'NR>1 && $0 !~ /^[[:space:]]*#/' "$CSV" | wc -l)
echo "===== $(date) JOB=${{SLURM_JOB_ID}} PARENT=${{SLURM_ARRAY_JOB_ID}} TASK=$IDX of 0..$((NUM_ROWS-1)) NODE=${{SLURMD_NODENAME:-unknown}} ====="

# Create metrics header if missing (dynamic from CSV header)
if [ ! -f logs/metrics.csv ]; then
  echo "job_id,task_id,node,start_epoch,$(head -n1 "$CSV" | tr -d '\\r'),maxrss_kb,elapsed_sec,cpu_pct,exit_code" >> logs/metrics.csv
fi

# Skip if index out of range
if [ "$IDX" -ge "$NUM_ROWS" ]; then
  echo "SKIP: No CSV row for index $IDX (data rows=$NUM_ROWS)"
  # Emit a SKIP line with empty CSV fields
  CSV_FIELDS=$(head -n1 "$CSV" | awk -F, '{{print NF}}')
  EMPTY_CSV=$(python - <<'PY'
n=int(input())
print(",".join([""]*n))
PY
<<< "$CSV_FIELDS")
  echo "$SLURM_JOB_ID,$IDX,${{SLURMD_NODENAME:-unknown}},$(date +%s),$EMPTY_CSV,0,0,0,SKIP" >> logs/metrics.csv
  exit 0
fi

# Grab the (IDX)-th data row (0-based), strip CRs
ROW=$(awk -F, 'NR>1 && $0 !~ /^[[:space:]]*#/' "$CSV" | sed -n "$((IDX+1))p" | tr -d '\\r')

# Parse header -> PARAMS[], row -> VALS[]
IFS=',' read -r -a PARAMS <<< "$(head -n1 "$CSV" | tr -d '\\r')"
IFS=',' read -r {" ".join(f"V{i}" for i in range(len(keys)))} <<< "$ROW"
VALS=({" ".join(f"$V{i}" for i in range(len(keys)))})

# Build CLI args: --key value for each CSV column
ARGS=()
for i in "${{!PARAMS[@]}}"; do
  ARGS+=( "--${{PARAMS[$i]}}" "${{VALS[$i]}}" )
done
# Always include job_id for your script
ARGS+=( "--job_id" "${{SLURM_ARRAY_JOB_ID}}" )

echo "Params: $ROW"
echo "Running: {pyfile} ${{ARGS[@]}}"

# Run and time it
TIMELOG="$(mktemp)"
set +e
/usr/bin/time -f "MAXRSS_KB=%M
ELAPSED_SEC=%e
CPU_PCT=%P
EXIT_CODE=%x" -o "$TIMELOG" \\
python {pyfile} "${{ARGS[@]}}"
STATUS=$?
set -e

# Parse /usr/bin/time output
MAXRSS_KB=""; ELAPSED_SEC=""; CPU_PCT=""; EXIT_CODE=""
while IFS='=' read -r K V; do
  case "$K" in
    MAXRSS_KB)   MAXRSS_KB="$V" ;;
    ELAPSED_SEC) ELAPSED_SEC="$V" ;;
    CPU_PCT)     CPU_PCT="$V" ;;
    EXIT_CODE)   EXIT_CODE="$V" ;;
  esac
done < "$TIMELOG"
rm -f "$TIMELOG"
[ -z "$EXIT_CODE" ] && EXIT_CODE="$STATUS"

# Append metrics (prepend job info, include full CSV row verbatim)
echo "$SLURM_JOB_ID,$IDX,${{SLURMD_NODENAME:-unknown}},$(date +%s),$ROW,$MAXRSS_KB,$ELAPSED_SEC,$CPU_PCT,$EXIT_CODE" >> logs/metrics.csv

echo "===== TASK $IDX DONE (MaxRSS=${{MAXRSS_KB}}KB, Elapsed=${{ELAPSED_SEC}}s, Exit=${{EXIT_CODE}}) ====="
"""

Path(slurm_name).write_text(slurm_text)
print(f"Wrote {slurm_name} (array=0-{len(rows)-1}) targeting: {pyfile}")
