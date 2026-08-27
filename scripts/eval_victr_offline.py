"""Offline held-out evaluation for the 3 VictrPolicy retrieval arms: mean
flow-matching loss on each task's 5% held-out test split (same splits.json
build_retrieval_pool.py wrote), retrieving with oracle query values and hard
top-k (viktr.retrieval.metrics.f_retrieve / retrieval.soft_topk with
training=False -- identical to fused_retrieve at this point).

Does NOT include the pi05 baseline (nyu-finger-robot/configs/yor-pi05-
ablation-*.yaml, jobs 15974326/15974327): that training run lives in a
different conda env against a different lerobot checkout
(/scratch/lim2045/lerobot-src) than this repo's own uv-managed
third_party/lerobot, so its held-out loss needs a separate eval invocation in
that environment -- this script only covers the 3 VictrPolicy arms trained
here.

Usage:
    uv run python scripts/eval_victr_offline.py \
        --pool-dir outputs/victr/icl_pool \
        --checkpoints vision=outputs/victr/vision/checkpoints/last \
                       value=outputs/victr/value/checkpoints/last \
                       vision+value=outputs/victr/vision_value/checkpoints/last \
        --out outputs/victr/offline_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn

from viktr.data.icl_dataset import (
    ACTION_CHUNK_SIZE,
    DEFAULT_REPO_ID,
    DEFAULT_ROOT,
    PRIMARY_CAMERA,
    TASKS,
    load_splits,
    robodopamine_value_at,
    task_slug,
)
from viktr.policy.pi05_context import VictrPolicy
from viktr.retrieval.chunk_dictionary import load_pools
from viktr.retrieval.embeddings import embed_frames, images_to_uint8_hwc, load_dinov2
from viktr.retrieval.fusion import ValueFusionMLP
from viktr.retrieval.metrics import f_retrieve
from viktr.retrieval.soft_topk import soft_fused_retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True, type=Path)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="metric=checkpoint_dir pairs, e.g. vision=outputs/victr/vision/checkpoints/last",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-context-chunks", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def eval_arm(metric: str, checkpoint_dir: Path, args, pools, dataset, meta, device) -> dict:
    policy = VictrPolicy.from_pretrained(checkpoint_dir / "pretrained_model")
    policy.to(device)
    policy.eval()

    fusion = None
    if metric == "vision+value":
        # training_state.pt sits next to pretrained_model/ inside the same checkpoint
        # dir (see train_victr.py's save_checkpoint).
        fusion = ValueFusionMLP().to(device)
        state = torch.load(checkpoint_dir.resolve() / "training_state.pt", map_location=device)
        fusion.load_state_dict(state["fusion"])
        fusion.eval()

    dinov2 = load_dinov2(device=device.type)
    preprocessor, _ = make_pi05_pre_post_processors(policy.config, dataset_stats=meta.stats)

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    total_loss, total_n = 0.0, 0
    for batch in dataloader:
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

        raw_primary = batch[PRIMARY_CAMERA].clone()
        episode_idx = batch["episode_index"].tolist()
        frame_idx = batch["frame_index"].tolist()
        tasks = batch["task"] if isinstance(batch["task"], list) else list(batch["task"])

        batch = preprocessor(batch)
        query_embeddings = embed_frames(images_to_uint8_hwc(raw_primary), model=dinov2)
        query_values = [robodopamine_value_at(e, f, root=args.root) for e, f in zip(episode_idx, frame_idx, strict=True)]

        context_chunks = []
        for i, task in enumerate(tasks):
            pool = pools[task]
            if metric == "vision":
                result = f_retrieve("vision", query_embeddings[i], None, pool, args.num_context_chunks)
            elif metric == "value":
                result = f_retrieve("value", query_embeddings[i], query_values[i], pool, args.num_context_chunks)
            else:
                result = soft_fused_retrieve(
                    query_embeddings[i], query_values[i], pool, args.num_context_chunks, fusion, tau=0.0, training=False
                )
            context_chunks.append(result.chunks)

        per_sample_loss, _ = policy.forward(batch, primary_camera=PRIMARY_CAMERA, context_chunks=context_chunks, reduction="none")
        total_loss += per_sample_loss.sum().item()
        total_n += per_sample_loss.shape[0]

    return {"metric": metric, "mean_loss": total_loss / total_n, "n_examples": total_n}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.pool_dir / "splits.json")
    pools = load_pools(args.pool_dir, TASKS, task_slug)

    test_episodes = sorted({e for task in TASKS for e in splits[task]["test"]})
    print(f"test episodes: {len(test_episodes)} across {len(TASKS)} tasks")
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    # See train_victr.py's matching comment: all 3 arms' checkpoints share the same
    # ACTION_CHUNK_SIZE (train_victr.py's config.chunk_size), so this is built directly
    # rather than re-loading a checkpoint's config just for its action_delta_indices.
    delta_timestamps = {"action": [i / meta.fps for i in range(ACTION_CHUNK_SIZE)]}
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=test_episodes, delta_timestamps=delta_timestamps)

    results = []
    for pair in args.checkpoints:
        metric, ckpt = pair.split("=", 1)
        print(f"evaluating {metric!r} from {ckpt}...")
        results.append(eval_arm(metric, Path(ckpt), args, pools, dataset, meta, device))
        print(f"  {results[-1]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
