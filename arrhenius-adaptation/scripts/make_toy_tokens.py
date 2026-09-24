#!/usr/bin/env python3
"""Generate synthetic pretraining-token shards for a trivial olmix smoke test.

olmo-core reads shards with `np.memmap(path, mode="r", dtype=dtype)` -- i.e. they
are raw, headerless flat arrays of token IDs, NOT real numpy .npy files despite
the .npy extension convention. `np.save()` would prepend a header and throw off
both file-size-based token counting (olmix/generate/synthesize_mixture.py) and
the memmap reader itself. So we write with `.tofile()`.

Usage:
    python make_toy_tokens.py --out-dir $OLMIX_ROOT/data --tokens-per-domain 200_000_000
"""

import argparse
from pathlib import Path

import numpy as np

# dolma2 tokenizer (olmo_core.data.tokenizer.TokenizerConfig.dolma2)
VOCAB_SIZE = 100278
EOS_TOKEN_ID = 100257

DOMAINS = ["web", "wiki", "code", "science"]

# Sequence length is fixed in olmix (olmix/model/transformer.py: SEQUENCE_LENGTH = 8192).
SEQ_LEN = 8192


def make_shard(path: Path, num_tokens: int, seed: int, doc_len_range: tuple[int, int] = (200, 800)) -> None:
    """Write one raw uint32 token shard with EOS-delimited pseudo-documents."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    tokens = np.empty(num_tokens, dtype=np.uint32)
    pos = 0
    while pos < num_tokens:
        doc_len = int(rng.integers(doc_len_range[0], doc_len_range[1]))
        doc_len = min(doc_len, num_tokens - pos)
        if doc_len <= 0:
            break
        # Avoid sampling the reserved EOS id as a "content" token.
        body_len = max(doc_len - 1, 0)
        body = rng.integers(0, EOS_TOKEN_ID, size=body_len, dtype=np.uint32)
        tokens[pos : pos + body_len] = body
        pos += body_len
        if pos < num_tokens:
            tokens[pos] = EOS_TOKEN_ID
            pos += 1

    tokens[: num_tokens].tofile(path)
    print(f"  wrote {path} ({num_tokens:,} tokens, {path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=200_000_000)
    ap.add_argument("--shards-per-domain", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert VOCAB_SIZE < 2**32
    assert args.tokens_per_domain >= SEQ_LEN * 4, "need several sequence lengths of data per domain"

    for i, domain in enumerate(DOMAINS):
        print(f"[{domain}]")
        per_shard = args.tokens_per_domain // args.shards_per_domain
        for s in range(args.shards_per_domain):
            shard_path = args.out_dir / domain / f"shard-{s:04d}.npy"
            make_shard(shard_path, per_shard, seed=args.seed * 1000 + i * 10 + s)

    total_gb = len(DOMAINS) * args.tokens_per_domain * 4 / 1e9
    print(f"\nDone. {len(DOMAINS)} domains x {args.tokens_per_domain:,} tokens each (~{total_gb:.2f} GB total).")


if __name__ == "__main__":
    main()
