from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_lewam import LeWAMConfig


PATCH_SIZE = 16


class LeWAMPolicy(PreTrainedPolicy):
    config_class = LeWAMConfig
    name = "lewam"

    def __init__(self, config: LeWAMConfig, dataset_stats: dict[str, Any] | None = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config

        from lewam.models.lewam import LeWAM

        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]

        self.lewam = LeWAM(
            model_dim=config.model_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_context_frames=config.num_context_frames,
            num_future_frames=config.num_future_frames,
            fps=config.fps,
            action_fps=config.action_fps,
            action_dim=action_dim,
            state_dim=state_dim,
            frame_latent_h=config.crop_size // PATCH_SIZE,
            frame_latent_w=(config.crop_size // PATCH_SIZE) * len(config.image_features),
            vlm_model_id=config.vlm_model_id,
            vlm_num_layers=config.vlm_num_layers,
            norm_strategy=config.norm_strategy,
            _pretrained_vlm=False,
            norm_stats=LeWAM._dummy_norm_stats(action_dim, state_dim),
        )

        self._camera_keys = sorted(config.image_features.keys())
        self._num_cameras = len(self._camera_keys)
        self._update_patch_grid()
        self.reset()

    def _update_patch_grid(self):
        patch_h = self.config.crop_size // PATCH_SIZE
        patch_w = patch_h * self._num_cameras
        if patch_h != self.lewam.frame_latent_h or patch_w != self.lewam.frame_latent_w:
            self.lewam.set_patch_grid(patch_h, patch_w)

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._frame_buffer = deque(maxlen=self.config.num_context_frames * 2)
        self._step_counter = 0

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    # ── Training ─────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        cam_frames = self._stack_camera_frames(batch)
        ctx_frames = cam_frames[:, :, :self.config.num_context_frames]
        fut_frames = cam_frames[:, :, self.config.num_context_frames:]

        context_tokens = self.lewam.encode_video(ctx_frames)
        future_tokens = self.lewam.encode_video(fut_frames)

        state = batch["observation.state"].squeeze(1)
        state = self.lewam.normalize_state(state)

        actions = batch["action"]
        dt = 1.0 / self.lewam.action_fps
        rel_velocity = (actions[:, 1:] - actions[:, :-1]) / dt
        rel_velocity = self.lewam.normalize_actions(rel_velocity)

        lang_tokens, lang_mask = self._encode_language(batch, ctx_frames)

        if lang_tokens is not None and self.config.lang_drop_rate > 0.0 and self.training:
            B = context_tokens.shape[0]
            drop = torch.rand(B, device=lang_tokens.device) < self.config.lang_drop_rate
            lang_tokens = lang_tokens.clone()
            lang_tokens[drop] = 0.0
            lang_mask = lang_mask.clone()
            lang_mask[drop] = True

        B = context_tokens.shape[0]
        t = torch.rand(B, device=context_tokens.device, dtype=context_tokens.dtype)

        x0_video = torch.randn_like(future_tokens)
        x_t_video = (1 - t[:, None, None]) * x0_video + t[:, None, None] * future_tokens

        x0_action = torch.randn_like(rel_velocity)
        x_t_action = (1 - t[:, None, None]) * x0_action + t[:, None, None] * rel_velocity

        video_vel, action_vel = self.lewam(
            x_t_video=x_t_video,
            x_t_action=x_t_action,
            context_tokens=context_tokens,
            t=t,
            state=state,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
        )

        target_video = future_tokens.detach() - x0_video
        target_action = rel_velocity - x0_action

        video_loss = F.mse_loss(video_vel, target_video)
        action_loss = F.mse_loss(action_vel, target_action)
        loss = video_loss + self.config.action_weight * action_loss

        return loss, {
            "video_loss": video_loss.item(),
            "action_loss": action_loss.item(),
        }

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        if self._step_counter % self.config.video_stride == 0:
            frame = torch.stack([batch[k] for k in self._camera_keys], dim=1)
            self._frame_buffer.append(frame)
        self._step_counter += 1

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            actions = actions[:, :self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()

        state = batch["observation.state"]
        if state.dim() == 3:
            state = state.squeeze(1)

        if self.config.test_circle:
            return self._generate_test_motion(state)

        frame = torch.stack([batch[k] for k in self._camera_keys], dim=1)
        self._frame_buffer.append(frame)

        frames = self._build_context_from_buffer()
        context_tokens = self.lewam.encode_video(frames)

        lang_tokens, lang_mask = None, None
        if self.lewam.vlm_encoder is not None and "task" in batch:
            last_frame = torch.cat([frames[:, i, -1] for i in range(self._num_cameras)], dim=-1)
            lang_tokens, lang_mask = self.lewam.encode_language(batch["task"], images=last_frame)

        norm_state = self.lewam.normalize_state(state)

        _, pred_actions = self.lewam.ode_solve(
            context_tokens, norm_state, lang_tokens, lang_mask,
            num_steps=self.config.num_ode_steps,
            smooth=self.config.smooth_actions,
        )

        rel_actions = self.lewam.unnormalize_actions(pred_actions)
        dt = 1.0 / self.config.action_fps
        abs_actions = state.unsqueeze(1) + torch.cumsum(rel_actions * dt, dim=1)
        return abs_actions

    def _generate_test_motion(self, state: Tensor) -> Tensor:
        B = state.shape[0]
        n = self.config.n_action_steps
        amplitude = 10.0
        period = self.config.action_fps * 2.0
        actions = state.unsqueeze(1).expand(B, n, -1).clone()
        t = torch.arange(n, device=state.device, dtype=state.dtype) + self._step_counter
        actions[:, :, 0] = state[:, 0:1] + amplitude * torch.sin(2.0 * torch.pi * t / period)
        self._step_counter += n
        return actions

    # ── Helpers ──────────────────────────────────────────────────────────

    def _stack_camera_frames(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.stack([batch[k] for k in self._camera_keys], dim=1)

    def _build_context_from_buffer(self) -> Tensor:
        if len(self._frame_buffer) == 0:
            raise RuntimeError("Frame buffer is empty, cannot predict without observations")
        n = self.config.num_context_frames
        frames = list(self._frame_buffer)[-n:]
        while len(frames) < n:
            frames.insert(0, frames[0])
        return torch.stack(frames, dim=2)

    def _encode_language(self, batch: dict[str, Tensor], ctx_frames: Tensor):
        if self.lewam.vlm_encoder is None or "task" not in batch:
            return None, None
        texts = batch["task"]
        last_ctx_frame = torch.cat([ctx_frames[:, i, -1] for i in range(self._num_cameras)], dim=-1)
        return self.lewam.encode_language(texts, images=last_ctx_frame)
