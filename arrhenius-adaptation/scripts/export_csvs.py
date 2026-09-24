#!/usr/bin/env python3
"""Export ratios.csv + metrics.csv from a completed olmix SLURM swarm.

olmix ships no CSV export tool (README step 4 assumes one exists). This
reconstructs both files from:
  - ratios: each variant's `mix` block (flattened colon-joined leaf -> weight)
  - metrics: the trainer's console log lines, of the form "  name=value",
    captured in the sbatch stdout (ConsoleLoggerCallback.log_metrics ->
    logging.info, which also carries every eval/downstream/* metric since
    they flow through the same trainer.record_metric() pipeline as W&B).

Schema required by olmix/fit/loaders.py (load_from_csv):
  - both files: `run,name,index,<data columns...>`
  - ratios domain columns must sum to ~1.0 per row (atol=0.01)
  - joined by `run`, ORDER PRESERVED PER FILE (not a positional merge) --
    this script writes both files pre-sorted by `run` to avoid mismatches.

Usage:
    python export_csvs.py --variants $OLMIX_ROOT/variants --logs $OLMIX_ROOT/logs \
        --out-dir $OLMIX_ROOT --metric-pattern 'eval/downstream/*'
"""

import argparse
import csv
import re
import subprocess
from pathlib import Path

import yaml

METRIC_LINE_RE = re.compile(r"^\s{4}([^=]+)=([\-0-9.eE+naninf]+)\s*$")


def load_variant_ratios(variants_dir: Path) -> dict[str, dict[str, float]]:
    """run/variant name -> {domain: weight}, read straight from each variant's mix: block."""
    ratios: dict[str, dict[str, float]] = {}
    for path in sorted(variants_dir.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        name = cfg["name"]
        mix = cfg.get("mix") or {}
        weights = {domain: float(entry["weight"]) for domain, entry in mix.items()}
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            print(f"  WARNING: {name} mix weights sum to {total:.4f}, not ~1.0")
        ratios[name] = weights
    return ratios


def parse_console_metrics(log_path: Path, metric_prefixes: list[str]) -> dict[str, float]:
    """Parse the LAST occurrence of each matching 'name=value' line in an sbatch .out log.

    ConsoleLoggerCallback prints a metrics block every metrics_log_interval steps;
    the last block before job completion holds the final/most-recent values.
    """
    if not log_path.exists():
        return {}
    latest: dict[str, float] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, raw_value = m.group(1).strip(), m.group(2)
        if metric_prefixes and not any(name.startswith(p) for p in metric_prefixes):
            continue
        try:
            latest[name] = float(raw_value)
        except ValueError:
            continue
    return latest


BARE_METRIC_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_. /()-]+?)=(-?[0-9.]+)\s*$")


def parse_wandb_binary_metrics(run_dir: Path) -> dict[str, float]:
    """Fallback: eval/downstream metrics are computed but NOT printed by
    ConsoleLoggerCallback's default filter (only train/optim/throughput/gpu_memory
    are) -- they only reach the offline wandb run's binary log. `strings` on that
    binary recovers the same 'name=value' text block wandb captured from stdout at
    eval time (confirmed present via `strings run-*.wandb | grep BPB`), even though
    it never appears in the sbatch .out file.
    """
    wandb_files = sorted(run_dir.glob("wandb/wandb/offline-run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime)
    if not wandb_files:
        return {}
    try:
        raw = subprocess.run(["strings", str(wandb_files[-1])], capture_output=True, text=True, check=True).stdout
    except Exception:
        return {}
    latest: dict[str, float] = {}
    for line in raw.splitlines():
        m = BARE_METRIC_LINE_RE.match(line)
        if not m:
            continue
        name, raw_value = m.group(1).strip(), m.group(2)
        if "BPB" not in name and "bpb" not in name:
            continue
        try:
            latest[name] = float(raw_value)
        except ValueError:
            continue
    return latest


def sanitize_metric_name(name: str) -> str:
    """eval/downstream/hellaswag_rc_5shot (BPB v2) -> hellaswag_rc_5shot_bpb_v2"""
    name = name.split("/")[-1]
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True, help="Dir of <job_name>-<jobid>.out sbatch logs")
    ap.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="checkpoints/slurm/<run>/ root -- fallback source for eval/downstream metrics "
        "(ConsoleLoggerCallback doesn't print them; only the offline wandb binary log has them)",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--metric-prefix",
        action="append",
        default=["eval/downstream/"],
        help="Only keep console metrics starting with this prefix (repeatable)",
    )
    args = ap.parse_args()

    ratios = load_variant_ratios(args.variants)
    if not ratios:
        raise SystemExit(f"No variant YAMLs found in {args.variants}")

    all_domains: set[str] = set()
    for weights in ratios.values():
        all_domains |= weights.keys()
    domains = sorted(all_domains)

    # Match each variant name to its most recent sbatch log (job name == variant name).
    metrics_by_run: dict[str, dict[str, float]] = {}
    all_metric_names: set[str] = set()
    for name in ratios:
        candidates = sorted(args.logs.glob(f"{name}-*.out"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"  WARNING: no log found for run '{name}' (looked for {name}-*.out in {args.logs})")
            continue
        raw = parse_console_metrics(candidates[-1], args.metric_prefix)
        if not raw and args.checkpoints:
            raw = parse_wandb_binary_metrics(args.checkpoints / name)
        clean = {sanitize_metric_name(k): v for k, v in raw.items()}
        metrics_by_run[name] = clean
        all_metric_names |= clean.keys()

    metric_names = sorted(all_metric_names)
    if not metric_names:
        print("  WARNING: no metrics parsed from any log (did you run with --no-eval?)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_names = sorted(ratios.keys())  # both files sorted identically by `run`

    ratios_path = args.out_dir / "ratios.csv"
    with open(ratios_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "name", "index", *domains])
        for idx, name in enumerate(run_names):
            weights = ratios[name]
            w.writerow([name, name, idx, *(weights.get(d, 0.0) for d in domains)])
    print(f"wrote {ratios_path} ({len(run_names)} runs x {len(domains)} domains)")

    metrics_path = args.out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "name", "index", *metric_names])
        for idx, name in enumerate(run_names):
            m = metrics_by_run.get(name, {})
            w.writerow([name, name, idx, *(m.get(k, "") for k in metric_names)])
    print(f"wrote {metrics_path} ({len(run_names)} runs x {len(metric_names)} metrics)")


if __name__ == "__main__":
    main()
