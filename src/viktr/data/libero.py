"""Adapts lerobot's `lerobot/libero` dataset into the per-episode arrays
viktr.retrieval.chunk_dictionary.build_chunk_dictionary expects.

`lerobot/libero` covers the 4 standard reproducibility suites (40 tasks); it's
the MVP demonstration source for building/testing retrieval before scaling to
full LIBERO-90+10 or the real bimanual dataset — same code path either way,
just a different repo_id / task list.
"""

from __future__ import annotations

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

DEFAULT_REPO_ID = "lerobot/libero"


def episodes_for_task(repo_id: str, task: str) -> list[int]:
    """Episode indices whose language instruction matches `task` exactly.

    lerobot's episode metadata doesn't carry a per-episode task column; task
    association only lives on the per-frame `task_index` column, so this reads
    that column (cheap: parquet only, no video decode) and takes the episodes
    that contain at least one matching frame.
    """
    meta = LeRobotDatasetMetadata(repo_id)
    task_index = meta.get_task_index(task)
    if task_index is None:
        raise ValueError(f"task {task!r} not found in {repo_id}; see meta.tasks for valid instructions")
    # download_videos=False: this only needs the parquet data columns, not the (much larger) video files.
    dataset = LeRobotDataset(repo_id, download_videos=False)
    table = dataset.select_columns(["episode_index", "task_index"])
    matches = {int(ep) for ep, ti in zip(table["episode_index"], table["task_index"], strict=True) if int(ti) == task_index}
    return sorted(matches)


def load_episode_arrays(
    repo_id: str,
    episode_indices: list[int],
    camera_keys: list[str] | None = None,
) -> list[dict]:
    """Returns one dict per episode: {"episode_index", "images": {cam: (T,H,W,3) uint8}, "proprio", "actions"}."""
    meta = LeRobotDatasetMetadata(repo_id)
    camera_keys = camera_keys or meta.camera_keys
    episode_indices = sorted(episode_indices)
    dataset = LeRobotDataset(repo_id, episodes=episode_indices)

    # dataset_from_index/dataset_to_index in meta.episodes are indices into the FULL dataset; once
    # LeRobotDataset is constructed with an `episodes` filter, its own indexing is local and
    # contiguous over just the selected episodes in ascending order, so bounds must be recomputed.
    lengths = {
        int(ep_idx): int(length)
        for ep_idx, length in zip(meta.episodes["episode_index"], meta.episodes["length"], strict=True)
    }

    out = []
    offset = 0
    for ep_idx in episode_indices:
        frm, to = offset, offset + lengths[ep_idx]
        offset = to
        images = {cam: [] for cam in camera_keys}
        proprio, actions = [], []
        for i in range(frm, to):
            item = dataset[i]
            for cam in camera_keys:
                frame = item[cam]
                arr = frame.permute(1, 2, 0).numpy() if hasattr(frame, "permute") else np.asarray(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                images[cam].append(arr)
            proprio.append(np.asarray(item["observation.state"], dtype=np.float32))
            actions.append(np.asarray(item["action"], dtype=np.float32))
        out.append(
            {
                "episode_index": ep_idx,
                "images": {cam: np.stack(frames, axis=0) for cam, frames in images.items()},
                "proprio": np.stack(proprio, axis=0),
                "actions": np.stack(actions, axis=0),
            }
        )
    return out
