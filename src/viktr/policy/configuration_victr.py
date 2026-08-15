"""VictrConfig: PI05Config extended with in-context retrieval parameters (Sec III-E).

Kept separate from `third_party/lerobot/src/lerobot/policies/pi05/configuration_pi05.py`
(not edited in place) so the vendored submodule stays a clean, diffable upstream
checkout — VictrConfig/VictrPolicy compose on top of it via subclassing instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

from viktr.retrieval.metrics import RetrievalMetric


@PreTrainedConfig.register_subclass("victr")
@dataclass
class VictrConfig(PI05Config):
    # k: number of retrieved chunks conditioning each query (paper Sec III-E).
    num_context_chunks: int = 1
    # L: frames per retrieved chunk. Should match the chunk_size used to build the
    # ChunkDictionary (viktr.retrieval.chunk_dictionary) the chunks came from.
    context_chunk_size: int = 10
    # K: frames subsampled per chunk (linspace(0, L-1, K), matching Robometer's
    # subsampling convention) and embedded through the shared SigLIP vision tower.
    context_frames_per_chunk: int = 1
    # Max token length of each chunk's "Task/State/Action" text summary.
    context_text_max_length: int = 64
    # Which f_retrieve mode (viktr.retrieval.metrics) selected the context chunks.
    # Informational here (retrieval itself happens upstream of the policy); wired
    # through so training/eval configs can log which variant is running.
    retrieval_metric: RetrievalMetric = "vision"

    def __post_init__(self):
        super().__post_init__()
        if self.num_context_chunks < 0:
            raise ValueError(f"num_context_chunks must be >= 0, got {self.num_context_chunks}")
        if self.context_frames_per_chunk < 1:
            raise ValueError(f"context_frames_per_chunk must be >= 1, got {self.context_frames_per_chunk}")
        if self.retrieval_metric not in ("vision", "value", "vision+value"):
            raise ValueError(f"unknown retrieval_metric: {self.retrieval_metric!r}")
