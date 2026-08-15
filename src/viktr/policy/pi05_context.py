"""IC-VLA: pi05 extended with retrieved-chunk context conditioning (paper Sec III-E).

Composes on top of `third_party/lerobot`'s `pi05` by subclassing rather than editing
the vendored submodule in place, so it stays a clean, diffable upstream checkout.

Mechanism: pi05's prefix (SigLIP image embeddings + tokenized "Task/State" text) is
attended to bidirectionally by the query's action-expert suffix via the block-causal
mask built by `make_att_2d_masks` (see vla_utils.py's docstring — a token's `att_mask`
bit of 1 starts a new causal block; 0 continues the previous block bidirectionally).
We prepend one such image+text block per retrieved chunk, ordered farthest-to-nearest,
each tagged with a learned neighbor-rank embedding and each starting a fresh causal
block. This gives: farthest chunk sees only itself; each nearer chunk also sees every
chunk before it; the query prefix (its own fresh block) sees every chunk; the query's
action-expert suffix sees everything. Context chunks never see the query, matching a
retrieval-conditioned (not leaking) design.
"""

from __future__ import annotations

from typing import Unpack

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import (
    make_att_2d_masks,
    prepare_attention_masks_4d,
    resize_with_pad_torch,
)
from lerobot.policies.pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    PI05Pytorch,
    get_gemma_config,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package

from viktr.policy.configuration_victr import VictrConfig
from viktr.retrieval.chunk_dictionary import Chunk

STATE_ACTION_BINS = 256  # matches Pi05PrepareStateTokenizerProcessorStep's digitization
PALIGEMMA_TOKENIZER_NAME = "google/paligemma-3b-pt-224"


def _digitize(values: np.ndarray) -> np.ndarray:
    """Clips to [-1, 1] and digitizes into STATE_ACTION_BINS bins, matching pi05's own
    state-text convention (Pi05PrepareStateTokenizerProcessorStep). Chunk proprio/actions
    are assumed pre-normalized to [-1, 1] the same way the base model's own observation.state
    is normalized upstream (see viktr.data.libero and the dataset's normalization stats) --
    they are clipped here so unnormalized inputs degrade gracefully instead of crashing."""
    clipped = np.clip(values, -1.0, 1.0)
    return np.digitize(clipped, bins=np.linspace(-1, 1, STATE_ACTION_BINS + 1)[:-1]) - 1


def chunk_to_context_text(chunk: Chunk) -> str:
    """Builds the "Task: ..., State: ...; Action: ..." text block for one retrieved chunk,
    summarizing what it demonstrated so the policy can read it like pi05 reads its own
    state (same digitization convention, applied to the chunk's first frame instead)."""
    state_bins = _digitize(chunk.proprio[0])
    state_str = " ".join(map(str, state_bins))
    num_action_frames = min(len(chunk.actions), 8)
    action_idx = np.linspace(0, len(chunk.actions) - 1, num_action_frames).round().astype(np.int64)
    action_bins = _digitize(chunk.actions[action_idx])
    action_str = "; ".join(" ".join(map(str, frame)) for frame in action_bins)
    cleaned_task = chunk.task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned_task}, State: {state_str}; Action: {action_str}"


class VictrPI05(PI05Pytorch):
    """PI05Pytorch + a learned per-neighbor-rank embedding for context blocks."""

    def __init__(self, config: VictrConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.config: VictrConfig = config
        width = get_gemma_config(config.paligemma_variant).width
        num_ranks = max(config.num_context_chunks, 1)
        self.neighbor_rank_embedding = nn.Embedding(num_ranks, width)

    def embed_context_chunks(
        self,
        context_images: list[list[Tensor]],
        context_img_masks: list[list[Tensor]],
        context_tokens: Tensor,
        context_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """context_images/img_masks: length-k lists (farthest-to-nearest), each element
        itself a list of per-frame (B, C, H, W) / (B,) tensors for that chunk (mirrors
        embed_prefix's `images`/`img_masks` argument shape, just one entry per chunk).
        context_tokens/context_masks: (B, k, L_text) tokenized "Task/State/Action" text.
        Returns the concatenated (embs, pad_masks, att_masks) for all k chunk blocks."""
        num_chunks = len(context_images)
        all_embs, all_pad_masks, all_att_masks = [], [], []
        for rank in range(num_chunks):
            embs, pad_masks, att_masks = self.embed_prefix(
                context_images[rank], context_img_masks[rank], context_tokens[:, rank], context_masks[:, rank]
            )
            att_masks = att_masks.clone()
            att_masks[:, 0] = True  # each chunk starts a fresh causal block (see module docstring)
            rank_ids = torch.full((embs.shape[0],), rank, device=embs.device, dtype=torch.long)
            embs = embs + self.neighbor_rank_embedding(rank_ids)[:, None, :].to(embs.dtype)
            all_embs.append(embs)
            all_pad_masks.append(pad_masks)
            all_att_masks.append(att_masks)
        return torch.cat(all_embs, dim=1), torch.cat(all_pad_masks, dim=1), torch.cat(all_att_masks, dim=1)

    def embed_prefix_with_context(
        self,
        images,
        img_masks,
        tokens,
        masks,
        context_images: list[list[Tensor]] | None = None,
        context_img_masks: list[list[Tensor]] | None = None,
        context_tokens: Tensor | None = None,
        context_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_embs, query_pad_masks, query_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        if not context_images:
            return query_embs, query_pad_masks, query_att_masks
        query_att_masks = query_att_masks.clone()
        query_att_masks[:, 0] = True  # query prefix also starts a fresh block, after all context chunks
        ctx_embs, ctx_pad_masks, ctx_att_masks = self.embed_context_chunks(
            context_images, context_img_masks, context_tokens, context_masks
        )
        embs = torch.cat([ctx_embs, query_embs], dim=1)
        pad_masks = torch.cat([ctx_pad_masks, query_pad_masks], dim=1)
        att_masks = torch.cat([ctx_att_masks, query_att_masks], dim=1)
        return embs, pad_masks, att_masks

    def forward(self, images, img_masks, tokens, masks, actions, noise, time, **context) -> Tensor:
        """Same as PI05Pytorch.forward, but the prefix includes context chunks (**context:
        context_images/context_img_masks/context_tokens/context_masks, see embed_prefix_with_context)."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_context(
            images, img_masks, tokens, masks, **context
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        context_images: list[list[Tensor]] | None = None,
        context_img_masks: list[list[Tensor]] | None = None,
        context_tokens: Tensor | None = None,
        context_masks: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Same as PI05Pytorch.sample_actions, prefix built with embed_prefix_with_context."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_with_context(
            images,
            img_masks,
            tokens,
            masks,
            context_images=context_images,
            context_img_masks=context_img_masks,
            context_tokens=context_tokens,
            context_masks=context_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        return euler_integrate(
            lambda input_x_t, current_timestep: self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=input_x_t,
                timestep=current_timestep,
            ),
            noise,
            num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )


class VictrPolicy(PI05Policy):
    """PI05Policy that conditions on retrieved Chunks. `predict_action_chunk`/`forward`
    take an extra `context_chunks: list[list[Chunk]]` argument: one ordered
    (farthest-to-nearest) list of up to `config.num_context_chunks` Chunk objects per
    batch element. Retrieval itself (f_retrieve) happens upstream; this class only
    embeds whatever chunks it's handed."""

    config_class = VictrConfig
    name = "victr"

    def __init__(self, config: VictrConfig, **kwargs):
        require_package("transformers", extra="pi")
        # Bypass PI05Policy.__init__ (it hardcodes PI05Pytorch as self.model) and
        # reproduce its body with VictrPI05 instead.
        from lerobot.policies.pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = VictrPI05(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self._context_tokenizer = None
        self.reset()

    @property
    def context_tokenizer(self):
        if self._context_tokenizer is None:
            from transformers import AutoTokenizer

            self._context_tokenizer = AutoTokenizer.from_pretrained(PALIGEMMA_TOKENIZER_NAME)
        return self._context_tokenizer

    def _prepare_context_frame(self, img: Tensor) -> Tensor:
        """(B, H, W, 3) float [0, 1] -> (B, 3, H, W) resized/padded + normalized to [-1, 1],
        matching PI05Policy._preprocess_images's channels-first output exactly (that method
        isn't reused directly since it dispatches on `self.config.image_features`, which
        context frames aren't registered under)."""
        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(img, *self.config.image_resolution)
        img = img * 2.0 - 1.0
        return img.permute(0, 3, 1, 2)

    def _prepare_context_images(self, context_chunks: list[list[Chunk]], primary_camera: str):
        """Returns (context_images, context_img_masks): length-k lists (farthest-to-nearest),
        each a length-K list of (B, C, H, W) / (B,) tensors — K = config.context_frames_per_chunk
        subsampled frames per chunk, preprocessed the same way as the query's own images."""
        device = next(self.parameters()).device
        num_chunks = self.config.num_context_chunks
        num_frames = self.config.context_frames_per_chunk

        context_images: list[list[Tensor]] = [[] for _ in range(num_chunks)]
        context_img_masks: list[list[Tensor]] = [[] for _ in range(num_chunks)]
        for rank in range(num_chunks):
            for frame_pos in range(num_frames):
                batch_frames = []
                for chunks in context_chunks:
                    chunk = chunks[rank]
                    frame_idx = np.linspace(0, len(chunk.proprio) - 1, num_frames).round().astype(np.int64)
                    batch_frames.append(chunk.images[primary_camera][frame_idx[frame_pos]])
                img = torch.from_numpy(np.stack(batch_frames, axis=0)).to(device)  # (B, H, W, 3) uint8
                img = img.float() / 255.0
                img = self._prepare_context_frame(img)
                mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)
                context_images[rank].append(img)
                context_img_masks[rank].append(mask)
        return context_images, context_img_masks

    def _prepare_context_text(self, context_chunks: list[list[Chunk]]) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        num_chunks = self.config.num_context_chunks
        texts = [chunk_to_context_text(chunks[rank]) for chunks in context_chunks for rank in range(num_chunks)]
        encoded = self.context_tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.context_text_max_length,
            return_tensors="pt",
        )
        batch_size = len(context_chunks)
        tokens = encoded["input_ids"].view(batch_size, num_chunks, -1).to(device)
        masks = encoded["attention_mask"].view(batch_size, num_chunks, -1).bool().to(device)
        return tokens, masks

    def _prepare_context(self, context_chunks: list[list[Chunk]] | None, primary_camera: str) -> dict:
        if not context_chunks:
            return {}
        for chunks in context_chunks:
            if len(chunks) != self.config.num_context_chunks:
                raise ValueError(
                    f"expected {self.config.num_context_chunks} context chunks per batch element, "
                    f"got {len(chunks)}"
                )
        context_images, context_img_masks = self._prepare_context_images(context_chunks, primary_camera)
        context_tokens, context_masks = self._prepare_context_text(context_chunks)
        return {
            "context_images": context_images,
            "context_img_masks": context_img_masks,
            "context_tokens": context_tokens,
            "context_masks": context_masks,
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        primary_camera: str,
        context_chunks: list[list[Chunk]] | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        context = self._prepare_context(context_chunks, primary_camera)

        actions = self.model.sample_actions(images, img_masks, tokens, masks, **context, **kwargs)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def forward(
        self,
        batch: dict[str, Tensor],
        primary_camera: str,
        context_chunks: list[list[Chunk]] | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict]:
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        context = self._prepare_context(context_chunks, primary_camera)

        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)

        losses = self.model.forward(images, img_masks, tokens, masks, actions, noise, time, **context)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {"loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist()}
        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict
