from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

NATIVE_FPS = 30


@PreTrainedConfig.register_subclass("lewam")
@dataclass
class LeWAMConfig(PreTrainedConfig):
    n_obs_steps: int = 1

    # TODO: lerobot factory passes CLI config (with defaults) into from_pretrained,
    # so the saved model config is never loaded. These defaults must match the
    # pretrained model until we fix config merging.
    model_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    vlm_model_id: str | None = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    vlm_num_layers: int = 4
    norm_strategy: str = "q1_q99"

    num_ode_steps: int = 2
    smooth_actions: bool = True

    crop_size: int = 224
    fps: float = 5.0
    action_fps: float = 30.0
    num_context_frames: int = 18
    num_future_frames: int = 8
    n_action_steps: int = 48

    action_weight: float = 1.0
    lang_drop_rate: float = 0.1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    @property
    def action_horizon(self) -> int:
        return int(self.num_future_frames / self.fps * self.action_fps)

    @property
    def video_stride(self) -> int:
        return round(NATIVE_FPS / self.fps)

    @property
    def observation_delta_indices(self) -> list[int]:
        stride = self.video_stride
        # context frames go backwards from 0, future frames go forward
        # e.g. 32 context + 8 future at stride 6: [-186, -180, ..., 0, 6, 12, ..., 42]
        context_indices = [-(self.num_context_frames - 1 - i) * stride for i in range(self.num_context_frames)]
        future_indices = [(i + 1) * stride for i in range(self.num_future_frames)]
        return context_indices + future_indices

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_horizon + 1))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=0.0, eps=1e-7)

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=50,
            num_decay_steps=50000,
            peak_lr=1e-4,
            decay_lr=0.0,
        )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("LeWAM requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("LeWAM requires an action feature.")
        if self.robot_state_feature is None:
            raise ValueError("LeWAM requires a state observation feature.")
