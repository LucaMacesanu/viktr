"""Robometer-based ValueEstimator — the MVP value signal (substitutes for a
trained IC-VFE per the user's decision).

Calling convention mirrors `lerobot.rewards.robometer` and
`projects/value-estimation/viewer/rewards/robometer_backend.py`: for a query
at frame t within an episode, score the K=4 subsampled window
`linspace(0, t, 4)` and take the model's progress output for that window.
"""

from __future__ import annotations

import numpy as np
import torch

from viktr.value.base import ValueEstimator

NUM_SUBSAMPLED_FRAMES = 4


class RobometerValueEstimator(ValueEstimator):
    def __init__(
        self,
        pretrained_path: str = "lerobot/Robometer-4B",
        image_key: str = "observation.images.image",
        device: str | None = None,
    ):
        from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
        from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel
        from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_key = image_key
        config = RobometerConfig(pretrained_path=pretrained_path, image_key=image_key)
        self.model = RobometerRewardModel.from_pretrained(pretrained_path, config=config)
        self.model.to(self.device).eval()
        self.encoder = RobometerEncoderProcessorStep(
            base_model_id=config.base_model_id,
            image_key=config.image_key,
            task_key=config.task_key,
            default_task=config.default_task,
            max_frames=NUM_SUBSAMPLED_FRAMES,
            use_multi_image=config.use_multi_image,
            use_per_frame_progress_token=config.use_per_frame_progress_token,
        )

    @torch.no_grad()
    def estimate(self, images: np.ndarray, instruction: str) -> float:
        """images: (T, H, W, 3) uint8 frames up to and including the current frame."""
        return float(self.estimate_episode(images, instruction)[-1])

    @torch.no_grad()
    def estimate_episode(self, images: np.ndarray, instruction: str, batch_size: int = 32) -> np.ndarray:
        from lerobot.lerobot_types import TransitionKey

        num_frames = len(images)
        windows = [np.linspace(0, t, NUM_SUBSAMPLED_FRAMES).round().astype(np.int64) for t in range(num_frames)]

        values = np.zeros(num_frames, dtype=np.float32)
        for start in range(0, num_frames, batch_size):
            batch_windows = windows[start : start + batch_size]
            stack = torch.from_numpy(np.stack([images[w] for w in batch_windows]))  # (B, K, H, W, 3) uint8
            transition = {
                TransitionKey.OBSERVATION: {self.image_key: stack},
                TransitionKey.COMPLEMENTARY_DATA: {"task": instruction},
            }
            obs = self.encoder(transition)[TransitionKey.OBSERVATION]
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in obs.items()}
            rewards = self.model.compute_reward(batch)
            values[start : start + len(batch_windows)] = rewards.float().cpu().numpy()
        return values
