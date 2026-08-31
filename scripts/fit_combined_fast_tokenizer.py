"""Fits one FAST action tokenizer jointly over the icl-dataset "expanded" training
set (viktr.data.icl_dataset.expanded_tasks(), 1,784 episodes -- same set
scripts/fit_expanded_fast_tokenizer.py fits alone, producing the
`yor-icl-expanded` tokenizer every `yor_icl_fast_victr_*`/`yor_icl_ki_*` config
currently uses) and the `adityx23/icl-demo-dataset` test set (285 episodes, 27
tasks, "cube tower stack" domain -- kept/successful subset only, 269 of 285).

Unlike fit_expanded_fast_tokenizer.py, normalization stats are NOT read from
either dataset's precomputed `dataset.meta.stats[ACTION]` -- icl-demo-dataset's
own meta/stats.json only covers its original split `action.left_ee` /
`action.right_ee` / `action.gripper` / `action.base_vel` / `action.lift_cmd`
columns, not the unified 20-dim `action` column added locally to make it
loadable via the same LeRobotDataset + process_episode pipeline as icl-dataset
(see notes -- the HF upload had an internal type inconsistency: action.lift_cmd
stored as a 1-element list while every other 1-dim feature in this dataset
family, e.g. `timestamp`, is a bare scalar, which crashes lerobot's v3 Arrow
schema cast; fixed on the local copy only, not pushed back to the Hub). More
fundamentally, quantile normalization must be fit over the union of both
datasets' raw actions, not a merge of two separately-computed quantile
summaries -- so this script collects raw (unnormalized) chunks from both
datasets first, concatenates them, then computes q01/q99 once over the
combined pool before calling train_fast_tokenizer.

Prerequisite (one-time, local-copy-only patch, already applied at
/scratch/lim2045/icl_ws/icl-demo-dataset): adds a unified `action` [20] column
(left_ee 7 | right_ee 7 | gripper 2 | base_vel 3 | lift_cmd 1, matching
icl-dataset's own `action` layout) and fixes `action.lift_cmd`'s list->scalar
type mismatch. Both changes are additive/corrective to the local download only.

Usage:
    uv run python scripts/fit_combined_fast_tokenizer.py \
        --icl-demo-root /scratch/lim2045/icl_ws/icl-demo-dataset \
        --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-expanded-plus-demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_train_tokenizer import (
    compute_compression_stats,
    process_episode,
    train_fast_tokenizer,
)

from viktr.data.icl_dataset import DEFAULT_ROOT, episodes_for_task, expanded_tasks


def collect_full_frame_actions(dataset: LeRobotDataset) -> np.ndarray:
    """Every frame's `action` vector for `dataset`'s (already episode-filtered)
    contents, via columnar access -- no video decode, no per-episode Python loop.
    Used only to compute normalization quantiles: a 10%-sampled action_horizon
    chunk pool (as collect_chunks below produces, meant for the actual DCT/BPE
    fit input) is too small a sample for q01/q99 on sparse-but-real dims like
    base_vel -- it spuriously collapsed to an exact-zero range in early testing,
    blowing up 2*(x-q01)/max(q99-q01,eps)-1 for any leftover nonzero outlier.
    Full per-frame data matches how lerobot's own dataset.meta.stats[ACTION] is
    computed (which icl-dataset's original solo fit implicitly relied on)."""
    return np.asarray(dataset.reader.hf_dataset["action"])


def collect_chunks(dataset: LeRobotDataset, episode_indices: list[int], action_horizon: int, sample_fraction: float) -> np.ndarray:
    all_chunks = []
    for i, ep_idx in enumerate(episode_indices):
        if i % 100 == 0:
            print(f"  episode {i}/{len(episode_indices)} (idx={ep_idx})...", flush=True)
        chunks = process_episode(
            (dataset, ep_idx, action_horizon, None, sample_fraction, "observation.state", False)
        )
        if chunks is not None:
            all_chunks.append(chunks)
    if not all_chunks:
        raise SystemExit("no action chunks collected -- check action_horizon vs. episode lengths")
    return np.concatenate(all_chunks, axis=0)


def demo_kept_successful_episodes(root: Path) -> list[int]:
    """Episode indices with keep=true & success=true, read directly from
    meta/episodes (single parquet file for this dataset's size -- no chunk glob
    needed, unlike icl-dataset's multi-shard meta/episodes)."""
    meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    kept = meta.loc[meta["keep"] & meta["success"], "episode_index"]
    return sorted(int(e) for e in kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icl-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--icl-demo-root", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=30)  # matches Pi0FastVictrConfig/Pi0KiConfig action_horizon
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10.0, help="DCT quantization scale.")
    parser.add_argument("--sample-fraction", type=float, default=0.1, help="Fraction of sliding-window action chunks per episode.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # --- icl-dataset (training set) ---
    tasks = expanded_tasks(root=args.icl_root)
    icl_episodes = sorted({ep for t in tasks for ep in episodes_for_task(t, root=args.icl_root)})
    print(f"[fit_combined] icl-dataset: {len(tasks)} tasks, {len(icl_episodes)} episodes")
    icl_dataset = LeRobotDataset(repo_id="icl-dataset", root=str(args.icl_root), episodes=icl_episodes)
    icl_chunks = collect_chunks(icl_dataset, icl_episodes, args.action_horizon, args.sample_fraction)
    print(f"[fit_combined] icl-dataset: {len(icl_chunks)} chunks, shape={icl_chunks.shape}")

    # --- icl-demo-dataset (test set) ---
    demo_episodes = demo_kept_successful_episodes(args.icl_demo_root)
    print(f"[fit_combined] icl-demo-dataset: {len(demo_episodes)} kept+successful episodes")
    demo_dataset = LeRobotDataset(repo_id="icl-demo-dataset", root=str(args.icl_demo_root), episodes=demo_episodes)
    demo_chunks = collect_chunks(demo_dataset, demo_episodes, args.action_horizon, args.sample_fraction)
    print(f"[fit_combined] icl-demo-dataset: {len(demo_chunks)} chunks, shape={demo_chunks.shape}")

    if icl_chunks.shape[-1] != demo_chunks.shape[-1]:
        raise SystemExit(f"action_dim mismatch: icl-dataset={icl_chunks.shape[-1]}, icl-demo-dataset={demo_chunks.shape[-1]}")

    all_chunks = np.concatenate([icl_chunks, demo_chunks], axis=0)
    print(f"[fit_combined] combined: {len(all_chunks)} chunks, shape={all_chunks.shape}")

    # Joint QUANTILES normalization, fit over full per-frame data from both
    # datasets (NOT the sampled chunk pool above -- see collect_full_frame_actions's
    # docstring for why: sparse-but-real dims like base_vel need the true
    # population's tails, which a 10%-chunk sample can miss entirely).
    icl_frames = collect_full_frame_actions(icl_dataset)
    demo_frames = collect_full_frame_actions(demo_dataset)
    all_frames = np.concatenate([icl_frames, demo_frames], axis=0)
    print(f"[fit_combined] combined full-frame pool for quantiles: {all_frames.shape}")
    q01 = np.quantile(all_frames, 0.01, axis=0)
    q99 = np.quantile(all_frames, 0.99, axis=0)
    denom = np.maximum(q99 - q01, 1e-8)
    normalized_chunks = 2.0 * (all_chunks - q01) / denom - 1.0

    tokenizer = train_fast_tokenizer(normalized_chunks, vocab_size=args.vocab_size, scale=args.scale)
    compression_stats = compute_compression_stats(tokenizer, normalized_chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "sources": [
            {
                "repo_id": "icl-dataset",
                "task_set": "expanded (viktr.data.icl_dataset.expanded_tasks())",
                "num_tasks": len(tasks),
                "num_episodes": len(icl_episodes),
                "num_chunks": len(icl_chunks),
            },
            {
                "repo_id": "adityx23/icl-demo-dataset",
                "task_set": "all kept+successful episodes",
                "num_episodes": len(demo_episodes),
                "num_chunks": len(demo_chunks),
            },
        ],
        "num_chunks": len(all_chunks),
        "action_horizon": args.action_horizon,
        "action_dim": int(all_chunks.shape[-1]),
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "sample_fraction": args.sample_fraction,
        "normalization_mode": "QUANTILES (q01/q99 fit jointly over full per-frame data from both datasets, not the sampled chunk pool used for the actual DCT/BPE fit, and not either dataset's own precomputed stats)",
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "compression_stats": compression_stats,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[fit_combined] Saved tokenizer + metadata to {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
