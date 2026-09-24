#!/usr/bin/env python3
"""Generate + submit SLURM sbatch scripts for an olmix variant directory.

This replaces `olmix launch run` (which hard-requires Beaker: it calls
`Beaker.from_env()` unconditionally, even under --dry-run). Instead we reuse the
two Beaker-FREE building blocks from olmix.launch.beaker:

  - mk_experiment_group(configs, group_uuid)  -- joins data.sources with mix
  - mk_instance_cmd(instance, config, group_id, beaker_user)  -- emits the exact
    argv olmix/launch/train.py expects (this is exactly what `olmix launch preview`
    prints; we generate the same thing straight into an sbatch script instead of
    stdout).

Usage:
    python olmix_slurm.py --variants $OLMIX_ROOT/variants --out-dir $OLMIX_ROOT/jobs \
        --account ehpc-dev-2026d08-289-gpu --partition gpu --time 00:30:00 \
        --venv $OLMIX_ROOT/venv --repo /home/nikolaos/olmix --work-root $OLMIX_ROOT \
        [--submit] [--no-eval]
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

# olmix must be importable -- run this with the built venv's python, or point
# PYTHONPATH at the repo directly.
from olmix.cli import _load_launch_configs  # type: ignore
from olmix.launch.beaker import mk_experiment_group, mk_instance_cmd  # type: ignore
from olmo_core.utils import generate_uuid  # type: ignore

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -A {account}
#SBATCH -p {partition}
#SBATCH --gres=gpu:1
#SBATCH -t {time}
#SBATCH -J {job_name}
#SBATCH -o {log_dir}/%x-%j.out
#SBATCH -e {log_dir}/%x-%j.err

source /etc/profile
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
set -euo pipefail
source {venv}/bin/activate

cd {repo}

# olmix/model/transformer.py hardcodes WandBCallback(enabled=True) with no config
# flag -- run offline instead of patching it. Point WANDB_DIR off the 30G home quota.
# WANDB_API_KEY is still required even in offline mode (checked before any network
# call is made) -- any non-empty dummy value satisfies the local writer.
export WANDB_MODE=offline
export WANDB_API_KEY=offline-local-run
export WANDB_DIR={work_root}/wandb
mkdir -p "$WANDB_DIR"

# flash-attn isn't installed (slow CUDA source build, skipped for this smoke test);
# olmix/model/transformer.py reads this to fall back to PyTorch's built-in SDPA.
export OLMIX_ATTN_BACKEND=torch

# On an unrecognized cluster name (any non-Ai2 cluster, e.g. "arrhenius"), olmix's
# checkpoint_dir otherwise defaults to s3://ai2-llm/... which this cluster can't
# write to (and the CheckpointerCallback 403s on it before training even starts).
# Route checkpoints under project storage instead of /tmp (node-local, wiped).
export OLMIX_LOCAL_ROOT={work_root}

# fit/generate/priors cache/output dirs are CWD-relative in olmix -- irrelevant
# here (this job only trains), but keep a sane CWD regardless.
export MPLBACKEND=Agg
export OMP_NUM_THREADS={cpus_per_task}

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

python {train_argv}

echo "=== done $(date) ==="
"""


def split_flag_value(fragment: str) -> list[str]:
    """mk_instance_cmd emits e.g. '-n foo' or '-s (...)' as ONE string per list
    element (fragment = flag + space + value). Split into argv-safe pieces."""
    fragment = fragment.strip()
    if fragment.startswith("--"):
        # long flags like --no-eval carry no value
        return [fragment]
    flag, _, value = fragment.partition(" ")
    return [flag, value] if value else [flag]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, required=True, help="Dir of generated variant YAMLs")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write sbatch scripts")
    ap.add_argument("--account", required=True)
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--time", default="00:30:00")
    ap.add_argument("--venv", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True, help="olmix repo root (train.py path is relative to this)")
    ap.add_argument("--work-root", type=Path, required=True, help="OLMIX_ROOT: logs/wandb/etc go under here")
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--no-eval", action="store_true", help="Force --no-eval regardless of config (Step 6 smoke test)")
    ap.add_argument("--limit", type=int, default=None, help="Only emit scripts for the first N variants")
    ap.add_argument("--submit", action="store_true", help="sbatch each script after writing it")
    args = ap.parse_args()

    log_dir = args.work_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = _load_launch_configs(str(args.variants))
    if args.limit:
        configs = configs[: args.limit]
    group_uuid = configs[0].group_id or generate_uuid()[:8]
    group = mk_experiment_group(configs, group_uuid)

    print(f"Loaded {len(configs)} variant(s), group_id={group_uuid}")

    for experiment, config in zip(group.instances, configs):
        argv = mk_instance_cmd(experiment, config, group.group_id, beaker_user="slurm")
        pieces: list[str] = []
        for fragment in argv:
            pieces.extend(split_flag_value(fragment))
        if args.no_eval and "--no-eval" not in pieces:
            pieces.append("--no-eval")
        train_argv = shlex.join(pieces)

        script = SBATCH_TEMPLATE.format(
            account=args.account,
            partition=args.partition,
            time=args.time,
            job_name=experiment.name,
            log_dir=log_dir,
            venv=args.venv,
            repo=args.repo,
            work_root=args.work_root,
            cpus_per_task=args.cpus_per_task,
            train_argv=train_argv,
        )
        script_path = args.out_dir / f"{experiment.name}.sbatch"
        script_path.write_text(script)
        script_path.chmod(0o755)
        print(f"  wrote {script_path}")

        if args.submit:
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            print(f"    {result.stdout.strip() or result.stderr.strip()}")
            if result.returncode != 0:
                print(f"    FAILED: {result.stderr}", file=sys.stderr)


if __name__ == "__main__":
    main()
