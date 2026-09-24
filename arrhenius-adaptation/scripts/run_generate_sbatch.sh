#!/bin/bash
# Generate SLURM sbatch scripts for olmix swarm variants.
# Run via sbatch (needs the venv, which is built for the GPU nodes' architecture):
#   sbatch -A <account> -p gpu --gres=gpu:1 -t 10 \
#     -J olmix-gensb -o logs/gensb-%j.out scripts/run_generate_sbatch.sh
#
# Extra args pass through to olmix_slurm.py, e.g. for a quick smoke test:
#   ... run_generate_sbatch.sh --no-eval --limit 1
source /etc/profile
module load "${OLMIX_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
set -euo pipefail

# OLMIX_ROOT holds the bulky artifacts (venv, tokenized data, checkpoints) and
# belongs on project storage, not home. WORK_ROOT is this directory, holding the
# scripts/configs and receiving the generated jobs/ and logs/.
OLMIX_ROOT="${OLMIX_ROOT:-/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work}"
WORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLMIX_REPO="$(cd "$WORK_ROOT/.." && pwd)"

source "$OLMIX_ROOT/venv/bin/activate"
cd "$WORK_ROOT"

python scripts/olmix_slurm.py \
  --variants variants \
  --out-dir jobs \
  --account "${OLMIX_SLURM_ACCOUNT:-ehpc-dev-2026d08-289-gpu}" \
  --partition "${OLMIX_SLURM_PARTITION:-gpu}" \
  --time "${OLMIX_SLURM_TIME:-00:30:00}" \
  --venv "$OLMIX_ROOT/venv" \
  --repo "$OLMIX_REPO" \
  --work-root "$OLMIX_ROOT" \
  "$@"
