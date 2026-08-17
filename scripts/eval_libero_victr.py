"""Task 6: LIBERO rollout/eval harness for VictrPolicy across retrieval-metric variants.

Runs closed-loop rollouts in a real LIBERO sim env (lerobot's `libero` env, via
`libero-project`/`robosuite`), retrieving context chunks from a `lerobot/libero`
demonstration pool with vision / value / vision+value / none (no-retrieval baseline)
and feeding them to VictrPolicy.predict_action_chunk at every action-chunk refill.

This validates the closed-loop path end-to-end; it is not a full published-protocol
benchmark run (see docs/source/libero.mdx for that -- 10 episodes/task across all 4
suites, hard resets). Defaults here are deliberately small so a first run finishes in
minutes; scale up via --n-episodes/--task-ids/--suite once the harness is confirmed
working. The single pi05_base-derived VictrPolicy, the DINOv2 embedder, and (if any
requested metric needs it) the Robometer value estimator are all loaded once and
reused across every task/metric in the run.

Usage:
  uv run python scripts/eval_libero_victr.py \
      --suite libero_object --task-ids 0 --n-episodes 2 \
      --retrieval-metrics vision,none
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import LiberoEnv as LiberoGymEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features

from viktr.data.libero import DEFAULT_REPO_ID, episodes_for_task, load_episode_arrays
from viktr.policy.configuration_victr import VictrConfig
from viktr.policy.pi05_context import VictrPolicy
from viktr.retrieval.chunk_dictionary import ChunkDictionary, build_chunk_dictionary
from viktr.retrieval.embeddings import embed_frames, load_dinov2
from viktr.retrieval.fusion import ValueFusionMLP
from viktr.retrieval.metrics import f_retrieve
from viktr.value.annotate import annotate_episode_values
from viktr.value.robometer import RobometerValueEstimator

ALL_METRICS = ("none", "vision", "value", "vision+value")


def _add_batch_dim(observation: dict) -> dict:
    """LiberoEnv (used here unvectorized, one env per task) returns unbatched leaf
    arrays, including inside the nested `robot_state` dict. preprocess_observation
    and LiberoProcessorStep add a batch dim to the final concatenated state, but
    LiberoProcessorStep._quat2axisangle requires its input already batched -- so a
    batch dim has to be added to every leaf (recursively) before that point."""
    out = {}
    for key, value in observation.items():
        if isinstance(value, dict):
            out[key] = _add_batch_dim(value)
        elif isinstance(value, np.ndarray):
            out[key] = value[None, ...]
        else:
            out[key] = value
    return out


@dataclass
class EpisodeResult:
    task_id: int
    metric: str
    episode: int
    success: bool
    steps: int
    total_reward: float


@dataclass
class TaskPool:
    dictionary: ChunkDictionary
    camera: str


def build_task_pool(
    task_text: str,
    camera: str,
    chunk_size: int,
    dinov2,
    needs_value: bool,
    value_estimator: RobometerValueEstimator | None,
) -> TaskPool:
    episode_indices = episodes_for_task(DEFAULT_REPO_ID, task_text)
    if not episode_indices:
        raise ValueError(f"no lerobot/libero demo episodes found for task {task_text!r}")
    episodes = load_episode_arrays(DEFAULT_REPO_ID, episode_indices, camera_keys=[camera])
    if needs_value:
        assert value_estimator is not None
        episodes = annotate_episode_values(episodes, value_estimator, task_text, camera)
    pool = build_chunk_dictionary(
        episodes, task=task_text, chunk_size=chunk_size, primary_camera=camera, embed_model=dinov2
    )
    return TaskPool(dictionary=pool, camera=camera)


def retrieve_context(
    metric: str,
    query_frame: np.ndarray,
    value_history: np.ndarray | None,
    task_text: str,
    pool: TaskPool,
    dinov2,
    fusion: ValueFusionMLP,
    value_estimator: RobometerValueEstimator | None,
    num_context_chunks: int,
) -> list | None:
    """query_frame: (H, W, 3) uint8, already flipped to match the dataset's orientation
    convention (see main()). Returns chunks ordered farthest-to-nearest (VictrPI05's
    expected context_chunks order), or None for the no-retrieval baseline."""
    if metric == "none":
        return None
    query_embedding = embed_frames(query_frame[None, ...], model=dinov2)[0]
    query_value = None
    if metric in ("value", "vision+value"):
        assert value_estimator is not None and value_history is not None
        query_value = float(value_estimator.estimate(value_history, task_text))
    result = f_retrieve(
        metric, query_embedding, query_value, pool.dictionary, k=num_context_chunks, fusion=fusion
    )
    return list(reversed(result.chunks))


def run_episode(
    policy: VictrPolicy,
    env: LiberoGymEnv,
    env_preprocessor: PolicyProcessorPipeline,
    env_postprocessor: PolicyProcessorPipeline,
    pre_processor: PolicyProcessorPipeline,
    post_processor: PolicyProcessorPipeline,
    task_text: str,
    camera: str,
    pool: TaskPool,
    metric: str,
    fusion: ValueFusionMLP,
    value_estimator: RobometerValueEstimator | None,
    num_context_chunks: int,
    n_action_steps: int,
    max_steps: int,
    dinov2,
    seed: int,
    device: torch.device,
) -> tuple[bool, int, float]:
    policy.reset()
    observation, info = env.reset(seed=seed)
    value_history: list[np.ndarray] = []

    action_queue: deque[torch.Tensor] = deque()
    total_reward = 0.0
    success = False
    step = 0
    while step < max_steps:
        if not action_queue:
            raw_frame = observation["pixels"][camera.rsplit(".", 1)[-1]][::-1, ::-1].copy()
            value_history.append(raw_frame)

            obs_t = preprocess_observation(_add_batch_dim(observation))
            obs_t["task"] = task_text
            obs_t = env_preprocessor(obs_t)
            obs_t = pre_processor(obs_t)

            value_hist_arr = np.stack(value_history, axis=0) if metric in ("value", "vision+value") else None
            context_chunks = retrieve_context(
                metric, raw_frame, value_hist_arr, task_text, pool, dinov2, fusion, value_estimator, num_context_chunks
            )
            with torch.no_grad():
                actions = policy.predict_action_chunk(
                    obs_t,
                    primary_camera=camera,
                    context_chunks=[context_chunks] if context_chunks else None,
                )
            for i in range(min(n_action_steps, actions.shape[1])):
                action_queue.append(post_processor(actions[:, i, :]))

        action = action_queue.popleft()
        action_transition = env_postprocessor({ACTION: action})
        action_np = action_transition[ACTION].to("cpu").numpy()[0]

        observation, reward, terminated, truncated, info = env.step(action_np)
        total_reward += float(reward)
        step += 1
        if info.get("is_success"):
            success = True
            break
        if terminated or truncated:
            break

    return success, step, total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default="0", help="comma-separated LIBERO task indices")
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument(
        "--retrieval-metrics", default="vision,none", help=f"comma-separated subset of {ALL_METRICS}"
    )
    parser.add_argument("--num-context-chunks", type=int, default=1)
    parser.add_argument("--context-chunk-size", type=int, default=10)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None, help="override the suite's default episode length")
    parser.add_argument("--hard-reset", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="optional path to write results JSON")
    args = parser.parse_args()

    task_ids = [int(t) for t in args.task_ids.split(",")]
    metrics = [m.strip() for m in args.retrieval_metrics.split(",")]
    for m in metrics:
        if m not in ALL_METRICS:
            raise ValueError(f"unknown retrieval metric {m!r}; choose from {ALL_METRICS}")
    needs_value = any(m in ("value", "vision+value") for m in metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(DEFAULT_REPO_ID)
    camera = meta.camera_keys[0]
    print(f"camera: {camera}, device: {device}")

    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    config = VictrConfig(
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        num_context_chunks=args.num_context_chunks,
        context_chunk_size=args.context_chunk_size,
        n_action_steps=args.n_action_steps,
    )
    print("loading VictrPolicy from lerobot/pi05_base (strict=False expected)...")
    policy = VictrPolicy.from_pretrained("lerobot/pi05_base", config=config, strict=False)
    policy.eval()
    pre_processor, post_processor = make_pi05_pre_post_processors(config, dataset_stats=meta.stats)
    env_preprocessor, env_postprocessor = LiberoEnvConfig(task=args.suite).get_env_processors()

    dinov2 = load_dinov2()
    fusion = ValueFusionMLP().to(device).eval()  # untrained -- see notes/progress.md
    value_estimator = RobometerValueEstimator(device=device.type) if needs_value else None

    suite = benchmark.get_benchmark_dict()[args.suite]()

    results: list[EpisodeResult] = []
    pool_cache: dict[int, TaskPool] = {}
    for task_id in task_ids:
        env = LiberoGymEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=args.suite,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            init_states=True,
            n_envs=1,
            control_mode="relative",
            hard_reset=args.hard_reset,
            episode_length=args.max_steps,
        )
        task_text = env.task_description
        print(f"\n=== task {task_id}: {task_text!r} ===")

        for metric in metrics:
            if metric != "none" and task_id not in pool_cache:
                pool_cache[task_id] = build_task_pool(
                    task_text, camera, args.context_chunk_size, dinov2, needs_value, value_estimator
                )
                print(f"  chunk pool built: {len(pool_cache[task_id].dictionary)} chunks")
            pool = pool_cache.get(task_id)

            for ep in range(args.n_episodes):
                seed = args.seed + ep
                success, steps, total_reward = run_episode(
                    policy=policy,
                    env=env,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    pre_processor=pre_processor,
                    post_processor=post_processor,
                    task_text=task_text,
                    camera=camera,
                    pool=pool,
                    metric=metric,
                    fusion=fusion,
                    value_estimator=value_estimator,
                    num_context_chunks=args.num_context_chunks,
                    n_action_steps=args.n_action_steps,
                    max_steps=env._max_episode_steps,
                    dinov2=dinov2,
                    seed=seed,
                    device=device,
                )
                print(f"  [{metric}] episode {ep}: success={success} steps={steps} reward={total_reward:.2f}")
                results.append(
                    EpisodeResult(
                        task_id=task_id, metric=metric, episode=ep, success=success, steps=steps, total_reward=total_reward
                    )
                )
        env.close()

    print("\n=== summary (success rate per task/metric) ===")
    for task_id in task_ids:
        for metric in metrics:
            rows = [r for r in results if r.task_id == task_id and r.metric == metric]
            if not rows:
                continue
            rate = sum(r.success for r in rows) / len(rows)
            print(f"  task {task_id} [{metric}]: {rate * 100:.0f}% ({sum(r.success for r in rows)}/{len(rows)})")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps([r.__dict__ for r in results], indent=2))
        print(f"\nwrote results to {out_path}")


if __name__ == "__main__":
    main()
