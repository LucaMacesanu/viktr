"""DO NOT USE FOR REAL TRAINING RUNS. Per project decision (2026-08-25), openpi
(third_party/openpi, JAX -- see openpi/training/config.py's yor_icl_* TrainConfigs)
is the sole training pipeline for pi05-based policies (VICTR/Pi05/KI or any future
arm); lerobot's native PyTorch pi05 training stack, which this script drives via
VictrPolicy (a lerobot PI05Policy subclass) + Accelerator, must not be used to train
pi05-based policies. This script (and its SLURM wrappers, victr_train_job.sh /
victr_cpu_extensive_smoke.sh / victr_smoke_job.sh) are kept only as a reference
implementation of the paper's soft-top-k vision+value fusion arm (retrieval/
soft_topk.py), which has no openpi equivalent yet -- porting it to openpi is
deferred. Non-training uses of the lerobot pi05 stack (eval_victr_offline.py,
eval_libero_victr.py) are unaffected by this restriction.

Trains VictrPolicy on icl-dataset's 4-task left-arm pnp subset (the same 4
tasks as nyu-finger-robot/configs/yor-pi05-ablation-*.yaml), one retrieval arm
at a time. See notes/vktr.pdf Sec III for the method and the approved plan
(cheerful-puzzling-metcalfe.md) for the design this implements:

  - Retrieval during training uses the RoboDopamine value model's real progress
    annotations (viktr.data.icl_dataset.robodopamine_value_at, meta/value_estimates/
    *.json), not Robometer/IC-VFE -- those are deployment-time-only per the paper
    (Sec III-F). The paper's own oracle t/(T-1) values (icl_dataset.oracle_value_at,
    Eq. 7) remain available as a fallback/ablation but aren't the default here.
  - "vision"/"value" arms have no learned retrieval params: plain hard top-k
    (retrieval.metrics.f_retrieve) against the task's precomputed pool.
  - "vision+value" jointly trains the fusion MLP via the soft top-k
    straight-through mechanism in retrieval.soft_topk (my concrete
    interpretation of the paper's Sec III-G description -- see that module's
    docstring for the exact design and why it scales the LOSS rather than
    patching gradients into pi05_context.py's embedding path).
  - No subtask cross-entropy loss (no subtask-label pipeline built), no MSA,
    no AWFM, no LoRA/KI partial-freeze -- flow-matching loss only, full
    finetune, matching the pi05 ablation configs for a fair comparison.

Usage:
    uv run python scripts/train_victr.py \
        --retrieval-metric vision \
        --pool-dir outputs/victr/icl_pool \
        --output-dir outputs/victr/vision \
        --steps 15000 --batch-size 128

Multi-GPU: launch via `accelerate launch --multi_gpu --num_processes=N ...` (mirrors
nyu-finger-robot/train.py's pattern for the pi05 ablation jobs). Single-process `python3
scripts/train_victr.py ...` still works unchanged -- Accelerator() is a no-op wrapper
in that case.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.feature_utils import dataset_to_policy_features

from viktr.data.icl_dataset import (
    ACTION_CHUNK_SIZE,
    CONTEXT_CHUNK_SIZE,
    DEFAULT_REPO_ID,
    DEFAULT_ROOT,
    PRIMARY_CAMERA,
    TASKS,
    load_splits,
    robodopamine_value_at,
    task_slug,
)
from viktr.policy.configuration_victr import VictrConfig
from viktr.policy.pi05_context import VictrPolicy
from viktr.retrieval.chunk_dictionary import ChunkDictionary, load_pools
from viktr.retrieval.embeddings import embed_frames, images_to_uint8_hwc, load_dinov2
from viktr.retrieval.fusion import ValueFusionMLP
from viktr.retrieval.metrics import f_retrieve
from viktr.retrieval.soft_topk import soft_fused_retrieve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-metric", required=True, choices=["vision", "value", "vision+value"])
    parser.add_argument("--pool-dir", required=True, type=Path, help="dir written by build_retrieval_pool.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-context-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--tau-start", type=float, default=1.0, help="soft top-k temperature at step 0 (vision+value only)")
    parser.add_argument("--tau-end", type=float, default=0.1, help="soft top-k temperature at the final step (vision+value only)")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def build_context(
    args: argparse.Namespace,
    tasks: list[str],
    query_embeddings: np.ndarray,
    query_values: list[float],
    pools: dict[str, ChunkDictionary],
    fusion: ValueFusionMLP | None,
    tau: float,
    training: bool,
) -> tuple[list[list], torch.Tensor]:
    """Per-example retrieval. Returns (context_chunks, per_example_weight) -- weight is
    exactly 1.0 in the forward value for every arm; only vision+value carries a real
    gradient through it (see retrieval/soft_topk.py)."""
    context_chunks = []
    weights = []
    for i, task in enumerate(tasks):
        pool = pools[task]
        if args.retrieval_metric == "vision":
            result = f_retrieve("vision", query_embeddings[i], None, pool, args.num_context_chunks)
            context_chunks.append(result.chunks)
            weights.append(torch.tensor(1.0))
        elif args.retrieval_metric == "value":
            result = f_retrieve("value", query_embeddings[i], query_values[i], pool, args.num_context_chunks)
            context_chunks.append(result.chunks)
            weights.append(torch.tensor(1.0))
        else:
            result = soft_fused_retrieve(
                query_embeddings[i], query_values[i], pool, args.num_context_chunks, fusion, tau, training
            )
            context_chunks.append(result.chunks)
            w = result.weights[0]
            for extra in result.weights[1:]:
                w = w * extra
            weights.append(w)
    return context_chunks, torch.stack(weights)


def main() -> None:
    args = parse_args()
    # find_unused_parameters=True: e.g. VictrPI05's neighbor_rank_embedding (and
    # potentially other retrieval-conditioned submodules) don't participate in every
    # forward pass depending on context_chunks/retrieval_metric, which crashed DDP's
    # default static bucket-reduction assumption on job 16082116's 2nd step
    # ("Expected to have finished reduction in the prior iteration...").
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main = accelerator.is_main_process
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    splits = load_splits(args.pool_dir / "splits.json")
    pools = load_pools(args.pool_dir, TASKS, task_slug)

    train_episodes = sorted({e for task in TASKS for e in splits[task]["train"]})
    if is_main:
        print(f"train episodes: {len(train_episodes)} across {len(TASKS)} tasks")

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    config = VictrConfig(
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        num_context_chunks=args.num_context_chunks,
        context_chunk_size=CONTEXT_CHUNK_SIZE,
        context_frames_per_chunk=1,
        retrieval_metric=args.retrieval_metric,
        chunk_size=ACTION_CHUNK_SIZE,
        n_action_steps=ACTION_CHUNK_SIZE,
        n_obs_steps=1,
        gradient_checkpointing=True,
        # PI05Config.dtype defaults to "float32" -- the paligemma+action-expert
        # backbone is constructed at this precision (modeling_pi05.py's
        # PaliGemmaWithExpertModel(..., precision=config.dtype, ...)). The
        # yor-pi05-ablation-*.yaml baseline jobs (same batch_size=128, n_gpus=2,
        # mem=256G) explicitly set dtype: bfloat16; leaving this at fp32 doubled
        # the backbone's memory footprint and OOM'd mid-backward on a 140G H200
        # (129.88 GiB in use, 10.06 GiB requested) on job 16080686's first step.
        dtype="bfloat16",
        # normalization_mapping/optimizer/scheduler fields all use PI05Config's
        # own defaults, matching yor-pi05-ablation-*.yaml (MEAN_STD, lr=5e-5,
        # warmup=2000, decay_steps=15000) for a fair comparison.
    )
    config.scheduler_decay_steps = args.steps

    # Without delta_timestamps, LeRobotDataset returns only the single frame at each
    # index -- the flow-matching suffix needs a full config.chunk_size-length action
    # horizon per sample (embed_suffix's att_masks is hardcoded to config.chunk_size,
    # so a mismatched actions length crashes make_att_2d_masks with a size mismatch).
    delta_timestamps = resolve_delta_timestamps(config, meta)
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=train_episodes, delta_timestamps=delta_timestamps)

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_fn,
    )

    if is_main:
        print("loading lerobot/pi05_base weights into VictrPolicy (strict=False expected)...")
    policy = VictrPolicy.from_pretrained("lerobot/pi05_base", config=config, strict=False)
    policy.to(device)
    policy.train()

    dinov2 = load_dinov2(device=device.type)
    fusion = None
    extra_params = []
    if args.retrieval_metric == "vision+value":
        fusion = ValueFusionMLP().to(device)
        extra_params = list(fusion.parameters())

    optimizer = config.get_optimizer_preset().build(list(policy.get_optim_params()) + extra_params)
    lr_scheduler = config.get_scheduler_preset().build(optimizer, args.steps)

    # accelerator.prepare wraps policy/fusion in DDP (no-op under single-process) and
    # makes dataloader/optimizer/lr_scheduler distributed-aware. Built the optimizer
    # from the raw (pre-wrap) modules' parameters above -- DDP wrapping preserves
    # parameter identity, so the optimizer's references stay valid after this.
    if fusion is not None:
        policy, fusion, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, fusion, optimizer, dataloader, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(policy, optimizer, dataloader, lr_scheduler)

    preprocessor, _postprocessor = make_pi05_pre_post_processors(config, dataset_stats=meta.stats)

    if args.wandb and is_main:
        import wandb

        wandb.init(project="viktr", name=f"victr-icl-{args.retrieval_metric}", config=vars(args))

    dl_iter = iter(dataloader)
    step = 0
    t_start = time.perf_counter()
    while step < args.steps:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

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

        tau = args.tau_start + (args.tau_end - args.tau_start) * min(step / max(args.steps - 1, 1), 1.0)
        context_chunks, weights = build_context(
            args, tasks, query_embeddings, query_values, pools, fusion, tau, training=True
        )
        weights = weights.to(device)

        # policy(...), not policy.forward(...): under DDP, calling .forward() directly
        # bypasses nn.Module.__call__ and skips the hooks DDP relies on for gradient sync.
        per_sample_loss, loss_dict = policy(batch, primary_camera=PRIMARY_CAMERA, context_chunks=context_chunks, reduction="none")
        loss = (per_sample_loss * weights).mean()

        optimizer.zero_grad()
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(
            list(policy.parameters()) + extra_params, config.optimizer_grad_clip_norm
        )
        optimizer.step()
        lr_scheduler.step()
        step += 1

        if args.log_freq > 0 and step % args.log_freq == 0 and is_main:
            elapsed = time.perf_counter() - t_start
            print(
                f"step {step}/{args.steps} loss={loss.item():.4f} grad_norm={float(grad_norm):.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} tau={tau:.3f} elapsed={elapsed:.1f}s"
            )
            if args.wandb:
                wandb.log({"loss": loss.item(), "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"], "tau": tau, **loss_dict}, step=step)

        if step % args.save_freq == 0 or step == args.steps:
            accelerator.wait_for_everyone()
            if is_main:
                unwrapped_fusion = accelerator.unwrap_model(fusion) if fusion is not None else None
                save_checkpoint(args.output_dir, step, accelerator.unwrap_model(policy), optimizer, lr_scheduler, unwrapped_fusion)

    if is_main:
        print("done.")


def save_checkpoint(output_dir: Path, step: int, policy, optimizer, scheduler, fusion) -> None:
    ckpt_dir = output_dir / "checkpoints" / f"{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir / "pretrained_model")
    state = {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    if fusion is not None:
        state["fusion"] = fusion.state_dict()
    torch.save(state, ckpt_dir / "training_state.pt")

    last_link = output_dir / "checkpoints" / "last"
    if last_link.is_symlink() or last_link.exists():
        last_link.unlink()
    last_link.symlink_to(ckpt_dir.name)
    print(f"saved checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
