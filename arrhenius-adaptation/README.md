# Running olmix on a SLURM cluster (Arrhenius / EuroHPC)

Olmix's `launch` step assumes Ai2's Beaker scheduler. This directory replaces it
with a SLURM equivalent so the full **Generate → Launch → Fit** workflow runs on
an ordinary HPC cluster with no Ai2 infrastructure.

Validated on Arrhenius (NVIDIA GH200, `aarch64`, SLURM): 8 swarm variants trained
end to end and fed into `olmix fit`, which produced a mixture proposal. The setup
should port to other SLURM + GPU clusters with changes confined to the account
name, partition name, and module names.

## What needed changing, and what didn't

Olmix has four stages. Only `launch` is tied to Ai2 infrastructure:

| Stage | Works as-is? | Notes |
|---|---|---|
| `olmix priors compute` | Yes | Pure local file stat + glob |
| `olmix generate` | Yes | Pure computation, no data I/O |
| `olmix launch run` | **No** | Calls `Beaker.from_env()` unconditionally |
| `olmix fit` | Yes | But nothing in olmix produces its input CSVs |

So this directory supplies two things olmix doesn't have: a SLURM launcher, and a
CSV exporter to bridge training output into `olmix fit`.

The launcher does **not** reimplement olmix's logic. It imports
`mk_experiment_group` and `mk_instance_cmd` from `olmix.launch.beaker` — these
two functions build the training command and never touch Beaker — and wraps their
output in an sbatch script instead of a Beaker job. This is the same code path
`olmix launch preview` uses.

## Contents

```
scripts/
  setup_env.sh             build the Python venv (run once, on a GPU node)
  make_toy_tokens.py       generate synthetic token shards for smoke-testing
  run_priors_generate.sh   olmix priors compute → generate → launch preview
  olmix_slurm.py           SLURM launcher; replaces `olmix launch run`
  run_generate_sbatch.sh   wrapper around olmix_slurm.py
  export_csvs.py           build ratios.csv + metrics.csv for `olmix fit`
configs/
  gen.yaml                 GenerationConfig: data sources + swarm sampling
  base.yaml                LaunchConfig template: model, eval tasks
  fit.yaml                 FitConfig: regression + proposer settings
examples/                  one sample of each generated artifact, for reference
```

Run artifacts (`logs/`, `jobs/`, `variants/`, `output/`, the CSVs) are gitignored;
they contain absolute paths and SLURM job IDs specific to one machine.

## Patches to olmix itself

Three changes outside this directory are required. They are additive and gated on
environment variables, so upstream behaviour is unchanged when the variables are
unset.

**`olmix/utils/cloud.py`** — allow local filesystem globs. Upstream raises
`NotImplementedError` for any path that isn't an `s3://` / `gs://` URL.

**`olmix/model/transformer.py`** — two independent additions:

- `OLMIX_ATTN_BACKEND` overrides the attention backend. Every model preset
  defaults to `flash_2`, which needs the `flash-attn` package (a slow CUDA
  source build). Set it to `torch` to use PyTorch's built-in SDPA.
- `OLMIX_LOCAL_ROOT` overrides the checkpoint directory. Upstream selects storage
  by matching the cluster name against Ai2's clusters (`jupiter`, `saturn`, …);
  anything else falls through to a default pointing at `s3://ai2-llm/...`, which
  an external cluster cannot write to.

## Environment setup

The venv must be built **on a GPU node**. On Arrhenius the login node is `x86_64`
while compute nodes are `aarch64`, so a venv built on the login node will not run.

```bash
sbatch -A <account> -p gpu --gres=gpu:1 -t 45 \
  -J olmix-setup -o logs/setup-%j.out \
  --wrap="bash scripts/setup_env.sh"
```

`setup_env.sh` encodes several version constraints that are easy to rediscover the
hard way:

- **`ai2-olmo-core`** is pinned in `pyproject.toml` to a git *branch* that no
  longer exists upstream. Install by the commit SHA recorded in `uv.lock`.
- **Python module name** — on Arrhenius the bare `Python/...` module resolves to
  an `x86_64` build even on a GH200 node; `GPU/Python/3.13.5-bare-gcc-2025b-eb`
  is the `aarch64` one. Check the equivalent on your cluster.
- **`beaker-py>=2.5.4`** plus `beaker-gantry` are needed to *import*
  `olmix.launch.beaker`, even though neither is ever called. Olmix's own pin
  (`>=1,<2`) is too old.
- **`wandb==0.19.9`**, not latest. The pinned olmo-core calls
  `wandb.finish(quiet=True)`; current wandb removed that argument. The failure
  mode is training completing successfully and then crashing during cleanup.
- **`protobuf`** — `wandb==0.19.9` needs `<6`, while `beaker-gantry`'s dependency
  chain pulls `google-cloud-storage` wanting `>=6.33.5`. Resolve by dropping
  `google-cloud-compute` (unused on this code path) and pinning
  `google-api-core<2.25` and `google-cloud-storage<2.19`.

## Storage layout

Keep bulk artifacts off the home filesystem — the venv alone is ~6 GB, and
checkpoints grow ~1.3 GB per variant. On Arrhenius the home quota is 30 GB.

```
<project storage>/olmix-work/
  venv/          Python environment
  data/          tokenized shards
  checkpoints/   per-run checkpoints and offline W&B data
```

## Adapting to another cluster

The scripts read these environment variables, each falling back to the Arrhenius
value. Set them rather than editing the scripts:

| Variable | Default | Purpose |
|---|---|---|
| `OLMIX_ROOT` | `/nobackup/proj/.../olmix-work` | Where venv, data and checkpoints live |
| `OLMIX_PYTHON_MODULE` | `GPU/Python/3.13.5-bare-gcc-2025b-eb` | `module load` target |
| `OLMIX_SLURM_ACCOUNT` | `ehpc-dev-2026d08-289-gpu` | `sbatch -A` |
| `OLMIX_SLURM_PARTITION` | `gpu` | `sbatch -p` |
| `OLMIX_SLURM_TIME` | `00:30:00` | Per-variant walltime |
| `OLMIX_EXPECT_ARCH` | `aarch64` | Guards against building the venv on the wrong node type; set empty to disable |

The repo path is derived from the scripts' own location, so cloning anywhere
works. Data paths in `configs/gen.yaml` and `configs/base.yaml` are absolute and
must be edited directly.

## Running the pipeline

Edit `configs/gen.yaml` (data sources, swarm size) and `configs/base.yaml` (model,
eval tasks) first, then:

```bash
# 1. Compute priors and sample swarm variants
sbatch -A <account> -p gpu --gres=gpu:1 -t 10 \
  -J olmix-gen -o logs/gen-%j.out scripts/run_priors_generate.sh

# 2. Generate one sbatch script per variant
sbatch -A <account> -p gpu --gres=gpu:1 -t 10 \
  -J olmix-gensb -o logs/gensb-%j.out scripts/run_generate_sbatch.sh

# 3. Submit them
for f in jobs/*.sbatch; do sbatch "$f"; done

# 4. After they finish: export CSVs and fit
python scripts/export_csvs.py --variants variants --logs logs \
  --checkpoints "$OLMIX_ROOT/checkpoints/slurm" --out-dir .
olmix fit --config configs/fit.yaml --output-dir output/my_fit
```

Step 4 needs the venv, so run it under sbatch on a GPU node too.

`run_generate_sbatch.sh` passes extra flags through to `olmix_slurm.py`; use
`--no-eval --limit 1` for a fast single-variant smoke test.

Results land in `output/my_fit/<hash>/`, including `*_optimal.json` with the
proposed mixture.

## Configuration for future runs

Four things are worth changing per experiment: the data, the eval tasks, the
model, and the swarm. Each is below with the file it lives in and the constraints
that will bite if ignored.

### 1. Data sources

In `configs/gen.yaml` **and** `configs/base.yaml` — both files carry a `data`
block and they must list the same sources. A source is a name plus one or more
glob patterns:

```yaml
data:
  sources:
    - name: web
      paths:
        - "/path/to/tokenized/web/*.npy"
    - name: code
      paths:
        - "/path/to/tokenized/code/*.npy"
        - "/path/to/tokenized/code_extra/*.npy"   # several globs are fine
```

The source names become the domain columns in `ratios.csv` and the keys of the
proposed mixture. Any number of sources works; cost scales with the number of
variants, not with the number of domains.

Sources may also nest into topics, which olmix then mixes hierarchically:

```yaml
    - name: dclm
      topics:
        - name: science
          paths: ["/path/to/dclm/science/*.npy"]
        - name: software
          paths: ["/path/to/dclm/software/*.npy"]
```

Nested topics produce colon-joined domain names (`dclm:science`). A source must
have exactly one of `paths`, `topics` or `quality` — the validator rejects
anything else. (`quality` splits a source by a quality score instead of by topic;
see `configs/examples/launch/quality_thresholds/` upstream.)

**Shards must be raw `uint32` token arrays**, not real `.npy` files — see Notes.
`scripts/make_toy_tokens.py` shows the expected format and is useful for a
synthetic smoke test before committing real data.

After changing sources, recompute the priors (`run_priors_generate.sh` prints
them) and paste the resulting `priors:` block into **both** `gen.yaml` and
`fit.yaml`. `olmix fit` asserts that `priors.relative_sizes` keys exactly match
the domain columns — a stale block fails there, late.

### 2. Evaluation tasks

In `configs/base.yaml`, under `eval.tasks`, grouped by family:

```yaml
eval:
  type: inloop
  tasks:
    qa:
      hellaswag_rc_5shot: eval/downstream/hellaswag_rc_5shot (BPB v2)
      arc_easy_test_rc_5shot: eval/downstream/arc_easy_test_rc_5shot (BPB v2)
    math:
      gsm8k_gold_bpb_5shot: eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)
```

The **key** is the olmo-core task ID passed to the trainer; the **value** is the
metric name that ends up as a `metrics.csv` column. Valid task IDs come from
`LABEL_TO_TASK_MAP` in `olmo_eval/tasks.py` (in the venv) — a wrong ID fails at
job start, not at submit time.

The family grouping matters only if you set
`regression.aggregate_task_families: true` in `fit.yaml`, which averages metrics
within each family before fitting. Upstream examples under
`configs/examples/launch/` show larger task sets.

Eval runs every `training.eval_interval` steps (default 1000) plus once at the
end. More tasks means more time per variant; for a first run keep it to a handful
of cheap ones.

### 3. Model size

In `configs/base.yaml`:

```yaml
training:
  proxy_model_id: olmo3_14m
  chinchilla_multiple: 0.5
  global_batch_size: 16
```

Valid `proxy_model_id` values are the keys of `MODEL_NUM_PARAMS` in
`olmix/aliases.py` — `olmo2_*` and `olmo3_*` from `1m` to `32b`. Token budget is
`20 × params × chinchilla_multiple`, so `olmo3_14m` at `0.5` trains on ~140M
tokens (~1000 steps, a few minutes per variant on one GH200).

**Keep `max_tokens` in `gen.yaml` equal to that number.** Olmix does not
cross-check the two files, and a mismatch shows up as mixtures whose token
budgets are impossible to satisfy.

The premise of the method is that a cheap proxy ranks mixtures the same way your
real target model would. Picking a proxy with your target's proportions, scaled
down, is more useful than picking the smallest model available.

For a size not in the list, add a factory entry in `_get_model_factory`
(`olmix/model/transformer.py`):

```python
"my_50m": lambda: TransformerConfig.llama_like(
    d_model=384, n_layers=8, n_heads=6, vocab_size=vocab_size, **kwargs
),
```

and a matching `"my_50m": 50_000_000` in `MODEL_NUM_PARAMS`. `llama_like` also
takes `n_kv_heads`, `head_dim`, sliding-window and RoPE settings; a
`llama_like_moe` variant exists for mixture-of-experts.

### 4. Swarm size and sampling

In `configs/gen.yaml`, under `swarm`:

```yaml
swarm:
  variants: 8          # number of mixtures to sample and train
  seed: 42
  min_strength: 0.1    # Dirichlet concentration range
  max_strength: 5.0
  minimum_weight: 0.002
```

`variants` is the main cost and quality knob — each variant is one training job.
It needs to exceed the number of domains or the regression is underdetermined.
Olmix has a guard for this in `fit/core.py`, but its condition compares a row
count against itself and so never fires — treat the requirement as yours to
enforce. Eight variants over four domains is enough to exercise the pipeline but
thin for a real fit; budget considerably more when the answer matters.

The Dirichlet concentration controls mixture shape. Low `min_strength` yields
sparse, peaked mixtures (a few domains dominate); high `max_strength` yields
mixtures close to the prior. Sampling sweeps the range, so keeping it wide gives
the regression varied points to fit.

Other useful fields on `SwarmConfig` (`olmix/aliases.py`):

| Field | Effect |
|---|---|
| `minimum_weight` | Zero out weights below this, avoiding unusably thin slices |
| `nonzero_weight` | Domains that must appear in every variant |
| `manual_prior` | Override the prior used as the Dirichlet centre |
| `mix_temperature` | Flatten the prior before sampling (values < 1.0) |
| `repetition_factor` | How many epochs over a domain are allowed |
| `enable_bound` | Cap weights by available tokens (leave on) |

### 5. Fit behaviour

In `configs/fit.yaml`:

```yaml
regression:
  type: log_linear              # log_linear | lightgbm | gp | autoscale | bimix | search
  aggregate_task_families: false
proposer:
  type: exact                   # exact | simulation | search
  kl_reg: 0.1                   # pull toward the prior; higher = more conservative
  fit_only: false               # true = diagnostics only, no mixture proposed
```

`log_linear` + `exact` is the default pairing and the only one validated here;
the `exact` proposer reads log-linear coefficients directly and will not work
with other regressors. Set `fit_only: true` to inspect regression quality before
trusting a proposal.

## Notes and gotchas

**Token shards are raw binary, not `.npy`.** Despite the extension, olmo-core
reads them with `np.memmap(path, dtype=...)`. Write them with `ndarray.tofile()`;
`np.save()` prepends a header and breaks both token counting and the reader.

**`MASTER_PORT` must be unique per job.** These are single-GPU jobs, and SLURM
will co-schedule several on one node (GH200 nodes have 4 GPUs). A fixed port
causes `EADDRINUSE` on all but the first. `olmix_slurm.py` derives it from
`$SLURM_JOB_ID`.

**Existing checkpoints silently skip training.** Resubmitting a variant that
already has a final-step checkpoint makes olmo-core load it, conclude training is
done, and exit without running eval — which surfaces later as an all-empty row in
`metrics.csv`. Delete `checkpoints/slurm/<run-name>/` to force a retrain.

**Eval metrics are not in the sbatch logs.** `ConsoleLoggerCallback` only prints
`train/*`, `optim/*`, `throughput/*` and `gpu_memory/*`. Downstream eval scores go
only to W&B, which runs offline here. `export_csvs.py` recovers them by reading
the offline W&B run's binary log, which is why `--checkpoints` is required
whenever eval is enabled.

**`olmix fit` fails unhelpfully on empty metrics.** If `metrics.csv` has no metric
columns, the proposer raises `ValueError: zero-size array ...` from inside cvxpy.
Check the CSV before debugging the solver.

