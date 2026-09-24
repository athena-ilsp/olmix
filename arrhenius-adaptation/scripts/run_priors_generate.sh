#!/bin/bash
# Compute token-count priors, sample swarm variants, and preview the training
# commands. Run via sbatch (needs the venv):
#   sbatch -A <account> -p gpu --gres=gpu:1 -t 10 \
#     -J olmix-gen -o logs/gen-%j.out scripts/run_priors_generate.sh
#
# Note: this deletes and regenerates variants/ — back it up first if you want to
# keep an existing swarm.
source /etc/profile
module load "${OLMIX_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
set -euo pipefail

OLMIX_ROOT="${OLMIX_ROOT:-/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work}"
WORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$OLMIX_ROOT/venv/bin/activate"
cd "$WORK_ROOT"

echo "=== priors compute ==="
olmix priors compute --config configs/gen.yaml -o configs/priors_computed.yaml
cat configs/priors_computed.yaml
echo
echo "NOTE: copy the priors: block above into configs/gen.yaml and configs/fit.yaml"
echo "      if the data sources changed since they were last written."

echo "=== generate ==="
rm -rf variants
olmix generate --config configs/gen.yaml --base configs/base.yaml --output variants/
ls -la variants/

# `launch preview` is the one launch code path that never constructs a Beaker
# client -- a useful sanity check that the variants produce valid train commands.
echo "=== launch preview ==="
olmix launch preview --variants variants/
