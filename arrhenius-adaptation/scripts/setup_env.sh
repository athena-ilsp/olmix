#!/bin/bash
# Build the olmix venv on an aarch64 GH200 node.
# Must run on a GPU node: the login node is x86_64 and its wheels are incompatible.

OLMIX_ROOT="${OLMIX_ROOT:-/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work}"
OLMIX_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# uv.lock pins this commit (olmo-core 2.4.0); the branch it came from was deleted upstream.
OLMO_CORE_SHA=58c0c38917716d1984610d8d6bd9ba977a028d0b

echo "=== node: $(hostname) arch=$(uname -m) ==="
# On Arrhenius the login node is x86_64 and the GPU nodes are aarch64, so a venv
# built on the wrong one silently produces unusable wheels. Adjust or drop this
# check on clusters where both node types share an architecture.
if [ -n "${OLMIX_EXPECT_ARCH:-aarch64}" ] && [ "$(uname -m)" != "${OLMIX_EXPECT_ARCH:-aarch64}" ]; then
    echo "FATAL: expected ${OLMIX_EXPECT_ARCH:-aarch64}, got $(uname -m)."
    echo "       Run this on a GPU node, or set OLMIX_EXPECT_ARCH= to skip."
    exit 1
fi

# The cluster's /etc/profile.d/debuginfod.sh references an unset var, so `set -u`
# must come AFTER sourcing profile/modules, not before.
source /etc/profile
# NOTE: the bare "Python/..." module resolves to an x86_64 (el9_epyc9005) build
# even when loaded from a GH200 node -- module names aren't arch-filtered here.
# The aarch64-native build lives under the GPU/ prefix (el9_gh200 tree).
module load "${OLMIX_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
set -eo pipefail
echo "python: $(python3 --version)"

# Keep pip's cache off the 30G home quota.
export PIP_CACHE_DIR=$OLMIX_ROOT/.pip-cache
export TMPDIR=$OLMIX_ROOT/.tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$OLMIX_ROOT"

rm -rf "$OLMIX_ROOT/venv"
python3 -m venv "$OLMIX_ROOT/venv"
source "$OLMIX_ROOT/venv/bin/activate"
pip install -q -U pip wheel setuptools

echo "=== torch (aarch64 manylinux_2_28 wheel) ==="
pip install torch
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"

echo "=== olmo-core @ pinned SHA (branch in pyproject.toml no longer exists) ==="
pip install "ai2-olmo-core[eval] @ git+https://github.com/allenai/OLMo-core.git@${OLMO_CORE_SHA}"

echo "=== olmix runtime deps ==="
# wandb pinned to 0.19.9, NOT latest: the pinned olmo-core commit's WandBCallback
# calls `wandb.finish(quiet=True)` on trainer cleanup (after training completes,
# confirmed live -- 1061/1061 steps ran fine, only the post-hoc finalize() call
# crashed). Current wandb (0.30.x) removed the `quiet` kwarg from finish();
# 0.19.9 is the newest release that still accepts it.
pip install click cvxpy ecos lightgbm matplotlib numpy pandas pydantic pyyaml \
            scikit-learn scipy seaborn statsmodels tqdm "wandb==0.19.9" yaspin \
            s3fs gcsfs boto3 "datasets>=3,<4" "pyarrow<21"

# beaker-py is required even for the Beaker-FREE mk_instance_cmd/mk_experiment_group
# helpers (used by our SLURM launcher): olmix/launch/beaker.py does
# `from beaker import Beaker` at module level, so importing anything from that
# file needs the package installed even though our code never calls it. It's a
# thin pure-Python API client -- installing it makes no network calls on its own.
# NOTE: olmix's own pyproject.toml pins ">=1,<2" but that is stale -- olmo-core's
# launch/beaker.py (which olmix/launch/beaker.py imports) needs symbols only present
# in beaker-py 2.x (e.g. BeakerImageNotFound), AND imports `gantry.api.GitRepoState`
# at module level. So beaker-py + beaker-gantry are needed just to IMPORT
# olmix.launch.beaker, even though our SLURM path never calls Beaker.from_env().
#
# google-cloud-compute is deliberately NOT installed. It is part of olmo-core's
# [beaker] extra but unused on this code path, and it forces protobuf>=6.33.5,
# which conflicts with wandb 0.19.9's protobuf<6. Installing it produces a venv
# where either wandb or google.rpc fails to import, depending which wins.
pip install "beaker-py>=2.5.4,<3.0" "GitPython>=3.0,<4.0" \
            "beaker-gantry>=3.4.3,<4.0"

# Hold the google stack below the protobuf 6 boundary. beaker-gantry pulls
# google-cloud-storage, whose recent releases require protobuf>=6.33.5; the
# generated google/rpc/*_pb2.py modules then refuse to load under protobuf 5.
# These pins match the versions verified working on Arrhenius.
pip install "google-api-core<2.25" "google-cloud-storage<2.19" "protobuf<6,>=3.19.5"

echo "=== olmix itself (--no-deps: olmo-core already satisfied above) ==="
cd "$OLMIX_REPO"
pip install -e . --no-deps

echo "=== verify ==="
python -c "import torch, olmo_core, olmix; print('imports ok')"
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok')"
# wandb and the google stack are the pair most likely to be broken by a resolver
# change (see the protobuf note above) -- check both, not just that olmix imports.
python -c "import wandb; print('wandb', wandb.__version__)"
python -c "from google.rpc import error_details_pb2; print('google protobuf ok')"
# The SLURM launcher imports these two; they pull in the whole beaker/gantry chain.
python -c "from olmix.launch.beaker import mk_experiment_group, mk_instance_cmd; print('launcher imports ok')"
olmix --help >/dev/null && echo "olmix CLI ok"
# fit is registered in a try/except ImportError, so absence means a dep failed silently
olmix fit --help >/dev/null && echo "olmix fit ok"
echo "=== SETUP COMPLETE ==="
