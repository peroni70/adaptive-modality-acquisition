#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o info_gain_diagnostic.out-%A
#SBATCH -t 04:00:00

# Site variations you may need:
#   --gres=gpu:volta:1     sites that require a GPU type
#   -p <partition>         sites without a usable default partition
#   --exclusive            when measuring wall-clock rather than correctness

# Does the info-gain value function learn anything, now that its loss no
# longer broadcasts (B, 1) logits against (B,) targets?
#
#   sbatch scripts/slurm/info_gain_diagnostic.sh
#   AMA_CONDA_ENV=myenv sbatch scripts/slurm/info_gain_diagnostic.sh
#
# Trailing arguments are passed to every stage, so the run can be scaled
# without editing the config:
#   sbatch scripts/slurm/info_gain_diagnostic.sh -o value_model.epochs=40
#
# Verdict is the skill_vs_constant field printed at the end:
#   ~0.00  no better than predicting the training mean - info_gain is dead
#   >0     it learned; how much says whether it is worth keeping

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

CONFIG=configs/patch_mnist_info_gain.yaml
EXTRA=("$@")

preflight || exit 1

# Fail fast from here on: the environment is known good.
set -euo pipefail

RUN_DIR=$(ama_run show "$CONFIG" "${EXTRA[@]}" --field run_dir)
echo "=== Config ==="
echo "  $CONFIG -> $RUN_DIR"
echo

echo "=== Stage 1/2: classifier ==="
time ama_run train-classifier "$CONFIG" "${EXTRA[@]}"

echo
echo "=== Stage 2/2: info-gain value model ==="
time ama_run train-value "$CONFIG" --value-fn info_gain "${EXTRA[@]}"

echo
echo "=== Verdict ==="
cat "$RUN_DIR/info_gain/value_model_scores.json"
