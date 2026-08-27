"""Fits a FAST action tokenizer on the 31-task expanded set (viktr.data.icl_dataset.
expanded_tasks(), Hub keep-field-curated, 1,784 episodes), for openpi's
yor_icl_ki_expanded_subtask config (openpi.models.pi0_ki.Pi0Ki).

The original KI tokenizer (third_party/nyu-finger-robot/outputs/fast_tokenizer/
yor-icl-pi05-easy-pnp-v2) was fit only on the 4-task pick-and-place subset's action
distribution via nyu-finger-robot/tools/fixes/fit_fast_tokenizer.py -- reusing it for
the expanded set's much more diverse tasks (bimanual passing, mallet strikes, sorting,
stacking, uncapping, ...) would quantize on bins calibrated for the wrong action
distribution. This script reuses that tool's exact core logic (same lerobot
lerobot_train_tokenizer internals: process_episode, apply_normalization,
train_fast_tokenizer, compute_compression_stats) but sources its episode list from
viktr.data.icl_dataset directly instead of nyu-finger-robot's train.py/resolve_episodes
-- that codebase's own filter_excluded predates the Hub keep-field fix (see
icl_dataset_v2_excluded_episodes.json), so going through it here would risk fitting on
a stale/inconsistent episode set. Runs entirely in viktr's own venv (which already
vendors the same lerobot fork nyu-finger-robot uses -- no cross-env dependency).

Usage:
    uv run python scripts/fit_expanded_fast_tokenizer.py \
        --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.configs import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_train_tokenizer import (
    apply_normalization,
    compute_compression_stats,
    process_episode,
    train_fast_tokenizer,
)
from lerobot.utils.constants import ACTION

from viktr.data.icl_dataset import DEFAULT_ROOT, episodes_for_task, expanded_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--action-horizon", type=int, default=30)  # matches pi0_ki.Pi0KiConfig(action_horizon=30)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10.0, help="DCT quantization scale.")
    parser.add_argument(
        "--sample-fraction", type=float, default=0.1, help="Fraction of sliding-window action chunks per episode."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = expanded_tasks(root=args.root)
    include_episodes = sorted({ep for t in tasks for ep in episodes_for_task(t, root=args.root)})
    print(f"[fit_expanded_fast_tokenizer] {len(tasks)} tasks, {len(include_episodes)} episodes")

    dataset = LeRobotDataset(repo_id="icl-dataset", root=str(args.root), episodes=include_episodes)
    print(f"[fit_expanded_fast_tokenizer] Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    all_chunks = []
    for i, ep_idx in enumerate(include_episodes):
        if i % 100 == 0:
            print(f"  episode {i}/{len(include_episodes)} (idx={ep_idx})...", flush=True)
        chunks = process_episode(
            (dataset, ep_idx, args.action_horizon, None, args.sample_fraction, "observation.state", False)
        )
        if chunks is not None:
            all_chunks.append(chunks)

    if not all_chunks:
        raise SystemExit("no action chunks collected -- check action_horizon vs. episode lengths")

    all_chunks = np.concatenate(all_chunks, axis=0)
    print(f"[fit_expanded_fast_tokenizer] Collected {len(all_chunks)} action chunks, shape={all_chunks.shape}")

    action_stats = dataset.meta.stats[ACTION]
    all_chunks = apply_normalization(all_chunks, action_stats, NormalizationMode.QUANTILES)

    tokenizer = train_fast_tokenizer(all_chunks, vocab_size=args.vocab_size, scale=args.scale)
    compression_stats = compute_compression_stats(tokenizer, all_chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "repo_id": "icl-dataset",
        "task_set": "expanded (viktr.data.icl_dataset.expanded_tasks())",
        "num_tasks": len(tasks),
        "num_episodes": dataset.num_episodes,
        "num_chunks": len(all_chunks),
        "action_horizon": args.action_horizon,
        "action_dim": int(all_chunks.shape[-1]),
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "sample_fraction": args.sample_fraction,
        "normalization_mode": "QUANTILES",
        "compression_stats": compression_stats,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[fit_expanded_fast_tokenizer] Saved tokenizer + metadata to {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
