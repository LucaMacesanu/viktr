"""Fits a FAST action tokenizer on the ICRA canonical action encoding (notes/
ICRA_plan.md Sec 1: delta EE position (relative to the chunk's own first frame) +
absolute rot6d orientation + gripper, per arm, 20-dim total), scoped to the trimmed
20-task/1,186-episode canonical extended set (assets/yor_icl_canonical_extended_
episodes.json), for arms 6 (RICL, pi0_fast_victr.Pi0FastVictrConfig) and 7
(pi0_fast.Pi0FASTConfig) of that plan.

Differs from scripts/fit_expanded_fast_tokenizer.py (the precedent this mirrors) in
two ways, both required for correctness, not just convention:
  1. That script tokenizes the RAW `action` column (quat+xyz+grip+base/lift) straight
     off the dataset row, quantile-normalized via lerobot's own dataset.meta.stats.
     This script transforms each sliding window into the canonical encoding first
     (yor_rotation.quat_wxyz_to_rot6d for orientation, delta-from-window's-own-first-
     frame for position -- exactly YorInputsCanonical/YorInputsDeltaRot6D's math) --
     a tokenizer fit on the old representation's distribution would quantize on the
     wrong bins for canonical's genuinely different action space.
  2. Normalizes using openpi's OWN already-computed canonical norm_stats
     (assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json's "actions"
     q01/q99), not a fresh quantile computation -- guarantees the tokenizer sees
     actions on the exact same normalized scale the trained model's own Normalize
     transform will use, rather than two independently-computed (and potentially
     slightly different) quantile estimates of nominally the same distribution.

Runs in openpi's venv (confirmed lerobot.scripts.lerobot_train_tokenizer is
importable there, and this needs openpi.policies.yor_rotation) -- NOT viktr's venv,
unlike fit_expanded_fast_tokenizer.py. Reads icl-dataset-fixed-obs's raw `action`
column directly via pandas (no LeRobotDataset object, no video decode needed --
action-only, same pattern already used by patch_pool_canonical.py this session).

Usage (from the viktr repo root, via openpi's venv):
    third_party/openpi/.venv/bin/python3 scripts/fit_canonical_fast_tokenizer.py \
        --output-dir third_party/nyu-finger-robot/outputs/fast_tokenizer/yor-icl-canonical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/lim2045/icl_ws/viktr/third_party/openpi/src")
from openpi.policies import yor_rotation  # noqa: E402

from lerobot.configs import NormalizationMode  # noqa: E402
from lerobot.scripts.lerobot_train_tokenizer import (  # noqa: E402
    apply_normalization,
    compute_compression_stats,
    train_fast_tokenizer,
)

ROOT = Path("/scratch/lim2045/icl_ws/icl-dataset-fixed-obs")
EPISODES_PATH = Path("/scratch/lim2045/icl_ws/viktr/third_party/openpi/assets/yor_icl_canonical_extended_episodes.json")
NORM_STATS_PATH = Path(
    "/scratch/lim2045/icl_ws/viktr/third_party/openpi/assets/yor_icl_pi05_canonical_extended/icl-dataset/norm_stats.json"
)

_LEFT_QUAT, _LEFT_POS = slice(0, 4), slice(4, 7)
_RIGHT_QUAT, _RIGHT_POS = slice(7, 11), slice(11, 14)
_LEFT_GRIP, _RIGHT_GRIP = slice(14, 15), slice(15, 16)


def canonical_action_window(action: np.ndarray) -> np.ndarray:
    """action: (action_horizon, 20) raw column window -> (action_horizon, 20) canonical
    [left_pos(3), left_rot6d(6), left_grip(1), right_pos(3), right_rot6d(6), right_grip(1)],
    delta relative to THIS window's own first row -- exactly YorInputsCanonical's math."""
    action = action.astype(np.float64)
    left_delta_pos = action[:, _LEFT_POS] - action[0:1, _LEFT_POS]
    right_delta_pos = action[:, _RIGHT_POS] - action[0:1, _RIGHT_POS]
    left_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[:, _LEFT_QUAT])
    right_rot6d = yor_rotation.quat_wxyz_to_rot6d(action[:, _RIGHT_QUAT])
    return np.concatenate(
        [left_delta_pos, left_rot6d, action[:, _LEFT_GRIP], right_delta_pos, right_rot6d, action[:, _RIGHT_GRIP]],
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-horizon", type=int, default=30)  # matches pi0_fast{,_victr} arms' action_horizon
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--sample-fraction", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    episodes = set(json.loads(EPISODES_PATH.read_text()))
    print(f"[fit_canonical_fast_tokenizer] {len(episodes)} episodes")

    norm_stats = json.loads(NORM_STATS_PATH.read_text())["norm_stats"]["actions"]
    q01 = np.asarray(norm_stats["q01"], dtype=np.float64)
    q99 = np.asarray(norm_stats["q99"], dtype=np.float64)
    print(f"[fit_canonical_fast_tokenizer] reusing canonical action norm_stats from {NORM_STATS_PATH}")

    cols = ["episode_index", "frame_index", "action"]
    parts = []
    for f in sorted((ROOT / "data").rglob("*.parquet")):
        part = pd.read_parquet(f, columns=cols)
        part = part[part["episode_index"].isin(episodes)]
        if len(part):
            parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    print(f"[fit_canonical_fast_tokenizer] loaded {len(df)} frames across {df['episode_index'].nunique()} episodes")

    all_chunks = []
    for i, (ep, g) in enumerate(df.groupby("episode_index")):
        if i % 200 == 0:
            print(f"  episode {i}/{len(episodes)} (idx={ep})...", flush=True)
        g = g.sort_values("frame_index")
        action = np.stack(g["action"].to_numpy())  # (T, 20) raw
        n = len(action)
        if n < args.action_horizon:
            continue

        n_windows = n - args.action_horizon + 1
        n_samples = max(1, int(n_windows * args.sample_fraction))
        rng = np.random.RandomState(hash(int(ep)) % (2**31))
        starts = rng.choice(n_windows, size=n_samples, replace=False)

        for s in starts:
            window = action[s : s + args.action_horizon]
            all_chunks.append(canonical_action_window(window))

    all_chunks = np.stack(all_chunks, axis=0)
    print(f"[fit_canonical_fast_tokenizer] collected {len(all_chunks)} canonical action chunks, shape={all_chunks.shape}")

    normalized = apply_normalization(all_chunks, {"q01": q01, "q99": q99}, NormalizationMode.QUANTILES)

    tokenizer = train_fast_tokenizer(normalized, vocab_size=args.vocab_size, scale=args.scale)
    compression_stats = compute_compression_stats(tokenizer, normalized)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "repo_id": "icl-dataset-fixed-obs",
        "task_set": "canonical trimmed 20-task set (assets/yor_icl_canonical_extended_episodes.json)",
        "num_episodes": len(episodes),
        "num_chunks": len(all_chunks),
        "action_horizon": args.action_horizon,
        "action_dim": int(all_chunks.shape[-1]),
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "sample_fraction": args.sample_fraction,
        "normalization_mode": "QUANTILES",
        "normalization_source": str(NORM_STATS_PATH),
        "compression_stats": compression_stats,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[fit_canonical_fast_tokenizer] saved tokenizer + metadata to {args.output_dir}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
