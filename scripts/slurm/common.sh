# Shared setup for batch jobs. Source this, do not execute it.
#
# Batch jobs do not inherit an interactive shell's conda state, and a
# `module load` can put a different interpreter ahead of yours on PATH. So
# pick the interpreter explicitly and prove it works before doing anything
# expensive - a job that dies in its first second should say why.
#
# Override the interpreter with either:
#   AMA_PYTHON=/path/to/python      sbatch ...
#   AMA_CONDA_ENV=myenv             sbatch ...

# SLURM may run the batch script from a spool copy, so the script's own path
# is not a reliable way back to the repo. Prefer the submission directory.
if [ -z "${AMA_ROOT:-}" ]; then
    AMA_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
fi
REPO_ROOT="$AMA_ROOT"

if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
    echo "ERROR: '$REPO_ROOT' does not look like the ama repo (no pyproject.toml)." >&2
    echo "  Submit from the repo root, or set AMA_ROOT=/path/to/adaptive-modality-acquisition." >&2
    return 1 2>/dev/null || exit 1
fi
cd "$REPO_ROOT" || exit 1

: "${AMA_CONDA_ENV:=ama}"

# Batch jobs invoke the interpreter directly rather than activating the
# environment, so set this here too: without it the job would pick up whatever
# is in ~/.local/lib/pythonX.Y/site-packages ahead of the environment.
export PYTHONNOUSERSITE=1

resolve_python() {
    if [ -n "${AMA_PYTHON:-}" ]; then
        echo "$AMA_PYTHON"
        return
    fi
    local env_python="$HOME/.conda/envs/$AMA_CONDA_ENV/bin/python3"
    if [ -x "$env_python" ]; then
        echo "$env_python"
        return
    fi
    command -v python3
}

PYTHON="$(resolve_python)"

preflight() {
    echo "=== Environment ==="
    echo "  repo        : $REPO_ROOT"
    echo "  interpreter : $PYTHON"
    echo "  host        : $(hostname)"

    if [ ! -x "$PYTHON" ]; then
        echo "ERROR: no usable interpreter at '$PYTHON'." >&2
        echo "  Set AMA_PYTHON=/path/to/python or AMA_CONDA_ENV=<env>." >&2
        return 1
    fi

    if ! "$PYTHON" -c "import ama" 2>/dev/null; then
        echo "ERROR: 'ama' is not importable by $PYTHON." >&2
        echo "  Install it into that environment:" >&2
        echo "    $PYTHON -m pip install -e $REPO_ROOT" >&2
        return 1
    fi

    "$PYTHON" - <<'PY'
import sys, torch
print(f"  torch       : {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"  GPU visible : {torch.cuda.is_available()}", end="")
print(f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")
if not (torch.__file__ or "").startswith(sys.prefix):
    print(f"  WARNING: torch resolves from outside the environment: {torch.__file__}",
          file=sys.stderr)
PY

    if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        echo "  WARNING: no GPU visible; this will run on CPU and take far longer." >&2
    fi
    echo
}

# Run the CLI through the module path, so the job does not depend on the
# console script being on PATH.
ama_run() {
    "$PYTHON" -m ama.cli "$@"
}
