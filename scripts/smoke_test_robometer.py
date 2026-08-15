"""Smoke test for the Robometer value estimator against one cached LIBERO episode.

Usage: uv run python scripts/smoke_test_robometer.py
"""

from __future__ import annotations

from viktr.data.libero import DEFAULT_REPO_ID, episodes_for_task, load_episode_arrays
from viktr.value.robometer import RobometerValueEstimator


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(DEFAULT_REPO_ID)
    task = meta.tasks.index[0]
    episode_indices = episodes_for_task(DEFAULT_REPO_ID, task)[:1]
    episodes = load_episode_arrays(DEFAULT_REPO_ID, episode_indices, camera_keys=[meta.camera_keys[0]])
    images = episodes[0]["images"][meta.camera_keys[0]]
    print(f"episode {episode_indices[0]}: {len(images)} frames, task={task!r}")

    estimator = RobometerValueEstimator(image_key=meta.camera_keys[0])
    values = estimator.estimate_episode(images, task)
    print("progress values (first 5, mid 5, last 5):")
    n = len(values)
    mid = n // 2
    print("  start:", values[:5])
    print("  mid:  ", values[mid - 2 : mid + 3])
    print("  end:  ", values[-5:])
    print(f"min={values.min():.3f} max={values.max():.3f} mean={values.mean():.3f}")


if __name__ == "__main__":
    main()
