#!/bin/bash
# Create the standard 'ama' environment and verify it can run the pipeline.
#
#   bash scripts/setup_env.sh              # create, or update if it exists
#   AMA_CONDA_ENV=other bash scripts/setup_env.sh
#
# Batch jobs default to the environment named here, so keeping the name
# consistent is what lets sbatch scripts work without configuration.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${AMA_CONDA_ENV:=ama}"

# A user site-packages directory (~/.local/lib/pythonX.Y/site-packages) is
# shared by every environment on the same Python version and sits ahead of the
# environment's own packages on sys.path. Left enabled, the environment
# silently inherits whatever is there - and pip treats those packages as
# already satisfied, so they never get installed here at all.
export PYTHONNOUSERSITE=1

# pip unpacks wheels through TMPDIR, and torch plus its CUDA dependencies run
# to several GB. On a cluster the default temp area is often home-backed and
# quota-limited, which surfaces as "OSError: [Errno 122] Disk quota exceeded"
# midway through the install. Prefer node-local scratch when it is available.
needs_scratch_tmpdir() {
    # Only redirect when TMPDIR is unset or points inside the home directory,
    # which is the case that hits a quota. A site-provided TMPDIR is left alone.
    [ -n "${AMA_KEEP_TMPDIR:-}" ] && return 1
    case "${TMPDIR:-}" in
        "") return 0 ;;
        "$HOME"/*) return 0 ;;
        *) return 1 ;;
    esac
}

if needs_scratch_tmpdir; then
    # Candidates in preference order. AMA_TMPDIR first, then locations various
    # clusters provide for node-local scratch, then plain /tmp. Entries that do
    # not exist are simply skipped, so this is harmless anywhere.
    for candidate in \
        "${AMA_TMPDIR:-}" \
        "/state/partition1/user/$USER" \
        "/scratch/$USER" \
        "/var/tmp/$USER" \
        "/tmp/$USER"
    do
        [ -z "$candidate" ] && continue
        if mkdir -p "$candidate" 2>/dev/null && [ -w "$candidate" ]; then
            export TMPDIR="$candidate"
            echo "Using TMPDIR=$TMPDIR for package installation."
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    echo "  On a cluster you may need: module load anaconda" >&2
    exit 1
fi

# Ask conda where an environment lives rather than guessing: a cluster's
# base install is often read-only, so a named environment lands in the user's
# envs directory instead of under the base prefix.
env_prefix() {
    conda info --envs | awk -v name="$AMA_CONDA_ENV" '$1 == name { print $NF; exit }'
}

ENV_PREFIX="$(env_prefix)"
if [ -n "$ENV_PREFIX" ]; then
    echo "Environment '$AMA_CONDA_ENV' exists at $ENV_PREFIX; updating."
else
    echo "Creating environment '$AMA_CONDA_ENV'..."
    conda create -y -n "$AMA_CONDA_ENV" python=3.12 pip
    ENV_PREFIX="$(env_prefix)"
fi

if [ -z "$ENV_PREFIX" ]; then
    echo "ERROR: conda did not report a prefix for '$AMA_CONDA_ENV' after creation." >&2
    conda info --envs >&2
    exit 1
fi

PYTHON="$ENV_PREFIX/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: no interpreter at $PYTHON after environment setup." >&2
    exit 1
fi

echo "Installing ama into $AMA_CONDA_ENV..."
"$PYTHON" -m pip install --upgrade pip

# Install torch before the package, pinned, from the PyTorch index:
#  - PyPI's default wheel tracks the newest release, which is not the
#    combination this code was tested against.
#  - The PyTorch index serves the CUDA builds that match cluster drivers.
# Set AMA_SKIP_TORCH=1 to manage torch yourself.
: "${AMA_TORCH_VERSION:=2.10.0}"
: "${AMA_TORCHVISION_VERSION:=0.25.0}"
: "${AMA_TORCH_INDEX:=https://download.pytorch.org/whl/cu128}"

if [ "${AMA_SKIP_TORCH:-0}" != "1" ]; then
    echo "Installing torch==$AMA_TORCH_VERSION from $AMA_TORCH_INDEX..."
    "$PYTHON" -m pip install --index-url "$AMA_TORCH_INDEX" \
        --retries 10 --resume-retries 20 \
        "torch==$AMA_TORCH_VERSION" "torchvision==$AMA_TORCHVISION_VERSION"
fi

"$PYTHON" -m pip install -e ".[dev,plots]"

# Persist the setting so an activated shell behaves like the batch jobs do.
conda env config vars set PYTHONNOUSERSITE=1 -n "$AMA_CONDA_ENV" >/dev/null

echo
echo "=== Verifying ==="
"$PYTHON" - "$ENV_PREFIX" <<'PY'
import sys
import torch, torchvision, sklearn, ama

prefix = sys.argv[1]
print(f"  ama         : {ama.__version__}")
print(f"  torch       : {torch.__version__} (CUDA build {torch.version.cuda})")
print(f"  GPU visible : {torch.cuda.is_available()}")

# Every dependency must resolve from inside this environment, or the
# environment is not reproducible on another machine.
stray = {
    name: mod.__file__
    for name, mod in (
        ("torch", torch), ("torchvision", torchvision), ("sklearn", sklearn)
    )
    if not (mod.__file__ or "").startswith(prefix)
}
if stray:
    print("  ERROR: these resolve from outside the environment:", file=sys.stderr)
    for name, path in stray.items():
        print(f"    {name}: {path}", file=sys.stderr)
    sys.exit(1)
print("  isolation   : ok (all dependencies inside the environment)")
PY
"$PYTHON" -m ama.cli --list-apps >/dev/null && echo "  CLI         : ok"

echo
echo "Done. Activate with:  conda activate $AMA_CONDA_ENV"
