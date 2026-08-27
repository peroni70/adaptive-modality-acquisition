#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o patch_mnist.out-%A-%a
#SBATCH -t 12:00:00
#SBATCH -a 0-1

# Site variations you may need:
#   --gres=gpu:volta:1     sites that require a GPU type
#   -p <partition>         sites without a usable default partition
#   --exclusive            recommended here: policy evaluation reports
#                          wall-clock time, which a shared node distorts

# One array task per value function.
#   sbatch scripts/slurm/patch_mnist.sh
#
# The classifier is shared across value functions, so train it once before
# submitting the array:
#   python -m ama.cli train-classifier configs/patch_mnist.yaml

# Load environment modules if this cluster uses them. Not needed to find the
# interpreter - that is resolved by absolute path below - but some sites
# require modules for drivers or MPI. Example:
#   AMA_MODULES="anaconda/2023a-pytorch" sbatch ...
if [ -n "${AMA_MODULES:-}" ]; then
    for _mod in $AMA_MODULES; do
        module load "$_mod" 2>/dev/null || echo "warning: could not load module $_mod" >&2
    done
fi

# Resolve the repo before sourcing: SLURM may run this from a spool copy.
AMA_ROOT="${AMA_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
source "$AMA_ROOT/scripts/slurm/common.sh"

CONFIG=configs/patch_mnist.yaml
VALUE_FNS=(acc_change bit_flip)
VALUE_FN=${VALUE_FNS[$SLURM_ARRAY_TASK_ID]}
EXTRA=("$@")

preflight || exit 1
set -euo pipefail

echo "Task $SLURM_ARRAY_TASK_ID -> value function $VALUE_FN"

ama_run greedy-order "$CONFIG" --value-fn "$VALUE_FN" "${EXTRA[@]}"
ama_run train-value  "$CONFIG" --value-fn "$VALUE_FN" "${EXTRA[@]}"
ama_run eval-policy  "$CONFIG" --value-fn "$VALUE_FN" "${EXTRA[@]}"
