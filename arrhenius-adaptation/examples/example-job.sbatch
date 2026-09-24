#!/bin/bash
#SBATCH -A ehpc-dev-2026d08-289-gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J toy-swarm-40f05e18-0000
#SBATCH -o /nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/logs/%x-%j.out
#SBATCH -e /nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/logs/%x-%j.err

source /etc/profile
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
set -euo pipefail
source /nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/venv/bin/activate

cd /home/nikolaos/olmix

# olmix/model/transformer.py hardcodes WandBCallback(enabled=True) with no config
# flag -- run offline instead of patching it. Point WANDB_DIR off the 30G home quota.
# WANDB_API_KEY is still required even in offline mode (checked before any network
# call is made) -- any non-empty dummy value satisfies the local writer.
export WANDB_MODE=offline
export WANDB_API_KEY=offline-local-run
export WANDB_DIR=/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/wandb
mkdir -p "$WANDB_DIR"

# flash-attn isn't installed (slow CUDA source build, skipped for this smoke test);
# olmix/model/transformer.py reads this to fall back to PyTorch's built-in SDPA.
export OLMIX_ATTN_BACKEND=torch

# On an unrecognized cluster name (any non-Ai2 cluster, e.g. "arrhenius"), olmix's
# checkpoint_dir otherwise defaults to s3://ai2-llm/... which this cluster can't
# write to (and the CheckpointerCallback 403s on it before training even starts).
# Route checkpoints under project storage instead of /tmp (node-local, wiped).
export OLMIX_LOCAL_ROOT=/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work

# fit/generate/priors cache/output dirs are CWD-relative in olmix -- irrelevant
# here (this job only trains), but keep a sane CWD regardless.
export MPLBACKEND=Agg
export OMP_NUM_THREADS=8

# Single-GPU, no torchrun: olmo-core's distributed init still expects these.
# (olmo_core.distributed.utils.validate_env_vars also requires NUM_NODES or
# LOCAL_WORLD_SIZE, and FS_LOCAL_RANK/LOCAL_RANK for a non-shared filesystem.)
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export NUM_NODES=1
export LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
# A hardcoded port collides when SLURM co-schedules two of these jobs on the same
# node (confirmed live: two variants landed on n531, second one hit EADDRINUSE on
# 29500). Derive a port from the job ID so concurrent single-node jobs don't clash.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "=== $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python olmix/launch/train.py train -n toy-swarm-40f05e18-0000 -g 40f05e18 -x 0.5 -S 42 -c arrhenius -u slurm -d uint32 -T dolma2 -m olmo3_14m -w False -y pretrain -b 4 -B 16 -E 1000 -e hellaswag_rc_5shot -e piqa_val_rc_5shot -e arc_easy_test_rc_5shot -e winogrande_val_rc_5shot -s '("wiki",["/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/data/wiki/*.npy"],0.444,1.0)' -s '("science",["/nobackup/proj/disk/ehpc-dev-2026d08-289/personal/nikolaos/olmix-work/data/science/*.npy"],0.556,1.0)'

echo "=== done $(date) ==="
