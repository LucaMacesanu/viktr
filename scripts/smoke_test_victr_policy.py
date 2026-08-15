"""Smoke test for VictrPolicy (task 5): one real forward pass of context-chunk-conditioned
pi05 (IC-VLA) on real lerobot/libero data, retrieving with the vision-only metric.

Loads real lerobot/pi05_base weights (strict=False: the new neighbor-rank embedding
has no pretrained counterpart) and predicts one action chunk conditioned on the single
nearest-neighbor retrieved chunk. Doesn't check action quality (the base model is
zero-shot on LIBERO and our context-conditioning weights are untrained) -- this only
validates that the whole path (retrieval -> chunk embedding -> block-causal masking ->
denoising loop) runs and produces a correctly-shaped action chunk.

Usage: uv run python scripts/smoke_test_victr_policy.py
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.feature_utils import dataset_to_policy_features

from viktr.data.libero import DEFAULT_REPO_ID, episodes_for_task, load_episode_arrays
from viktr.policy.configuration_victr import VictrConfig
from viktr.policy.pi05_context import VictrPolicy
from viktr.retrieval.chunk_dictionary import build_chunk_dictionary
from viktr.retrieval.embeddings import embed_frames, load_dinov2
from viktr.retrieval.metrics import vision_retrieve

CONTEXT_CHUNK_SIZE = 10


def main() -> None:
    meta = LeRobotDatasetMetadata(DEFAULT_REPO_ID)
    task = meta.tasks.index[0]
    camera = meta.camera_keys[0]
    print(f"task: {task!r}, camera: {camera}")

    episode_indices = episodes_for_task(DEFAULT_REPO_ID, task)[:3]
    episodes = load_episode_arrays(DEFAULT_REPO_ID, episode_indices, camera_keys=meta.camera_keys)
    query_episode, *pool_episodes = episodes
    print(f"query episode {query_episode['episode_index']} ({len(query_episode['proprio'])} frames); "
          f"pool episodes: {[e['episode_index'] for e in pool_episodes]}")

    dinov2 = load_dinov2()
    pool = build_chunk_dictionary(
        pool_episodes, task=task, chunk_size=CONTEXT_CHUNK_SIZE, primary_camera=camera, embed_model=dinov2
    )
    print(f"chunk pool size: {len(pool)}")

    query_frame_idx = len(query_episode["proprio"]) // 3
    query_embedding = embed_frames(query_episode["images"][camera][query_frame_idx : query_frame_idx + 1], model=dinov2)[0]
    retrieved = vision_retrieve(query_embedding, pool, k=1)
    nearest_chunk = retrieved.chunks[0]
    print(
        f"query frame {query_frame_idx}: nearest chunk from episode {nearest_chunk.episode_index} "
        f"frames [{nearest_chunk.start_frame},{nearest_chunk.end_frame}) (dist={retrieved.scores[0]:.4f})"
    )

    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    config = VictrConfig(
        input_features=input_features,
        output_features=output_features,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_context_chunks=1,
        context_chunk_size=CONTEXT_CHUNK_SIZE,
        context_frames_per_chunk=1,
        retrieval_metric="vision",
    )

    print("loading lerobot/pi05_base weights into VictrPolicy (strict=False expected: "
          "neighbor_rank_embedding has no pretrained counterpart)...")
    policy = VictrPolicy.from_pretrained("lerobot/pi05_base", config=config, strict=False)
    policy.eval()

    pre_processor, _post_processor = make_pi05_pre_post_processors(config, dataset_stats=meta.stats)

    raw_obs = {
        "observation.state": query_episode["proprio"][query_frame_idx],
        camera: query_episode["images"][camera][query_frame_idx],
    }
    for cam in meta.camera_keys:
        if cam not in raw_obs:
            raw_obs[cam] = query_episode["images"][cam][query_frame_idx]

    device = torch.device(config.device)
    obs = prepare_observation_for_inference(raw_obs, device, task=task)
    batch = pre_processor(obs)

    with torch.no_grad():
        actions = policy.predict_action_chunk(batch, primary_camera=camera, context_chunks=[[nearest_chunk]])
    print(f"predicted action chunk shape: {tuple(actions.shape)}")
    assert actions.shape[0] == 1
    assert actions.shape[1] == config.chunk_size
    assert actions.shape[2] == config.output_features["action"].shape[0]
    assert torch.isfinite(actions).all()

    no_context_actions = policy.predict_action_chunk(batch, primary_camera=camera, context_chunks=None)
    print(f"no-context action chunk shape: {tuple(no_context_actions.shape)}")
    assert not torch.allclose(actions, no_context_actions), "context conditioning had no effect on output"

    print("\nOK: VictrPolicy forward pass with real retrieved context ran end-to-end.")


if __name__ == "__main__":
    main()
