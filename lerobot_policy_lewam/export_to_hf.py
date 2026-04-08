"""
Convert a LeWAM .pt checkpoint to HuggingFace format for use with lerobot.

This creates a directory with config.json + model.safetensors that can be
loaded via LeWAMPolicy.from_pretrained() or pushed to the Hub.

Usage:
    source .venv/bin/activate
    python -m lerobot_policy_lewam.export_to_hf \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/output \
        --cameras image1 image2 \
        --repo-id ehalicki/lewam-so101  # optional, pushes to Hub
"""

import argparse
from pathlib import Path


def export(
    checkpoint_path: str,
    output_dir: str | Path,
    camera_names: list[str],
    repo_id: str | None = None,
):
    from lewam.models.lewam import LeWAM
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot_policy_lewam.configuration_lewam import LeWAMConfig
    from lerobot_policy_lewam.modeling_lewam import LeWAMPolicy

    print(f"Loading checkpoint from {checkpoint_path}...")
    raw_model = LeWAM.from_checkpoint(checkpoint_path)
    cfg = raw_model.config

    input_features = {}
    for cam in camera_names:
        input_features[f"observation.images.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640),
        )
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(cfg["action_dim"],),
    )

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION, shape=(cfg["action_dim"],),
        ),
    }

    lewam_config = LeWAMConfig(
        input_features=input_features,
        output_features=output_features,
        model_dim=cfg["model_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        vlm_model_id=cfg.get("vlm_model_id"),
        vlm_num_layers=cfg.get("vlm_num_layers", 8),
        norm_strategy=cfg.get("norm_strategy", "q1_q99"),
        num_context_frames=cfg["num_context_frames"],
        num_future_frames=cfg["num_future_frames"],
        fps=cfg["fps"],
        action_fps=cfg["action_fps"],
        crop_size=raw_model.frame_latent_h * LeWAM.VJEPA_PATCH_SIZE,
        num_ode_steps=10,
        smooth_actions=True,
    )

    print("Building LeWAMPolicy...")
    policy = LeWAMPolicy(lewam_config)
    policy.lewam.load_state_dict(raw_model.state_dict())

    output_dir = Path(output_dir)
    print(f"Saving HuggingFace format to {output_dir}...")
    policy.save_pretrained(output_dir)

    from lerobot_policy_lewam.processor_lewam import make_lewam_pre_post_processors
    preprocessor, postprocessor = make_lewam_pre_post_processors(lewam_config)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    card = f"""---
library_name: lerobot
tags:
- lewam
- robotics
- flow-matching
---

# LeWAM

Joint video-action flow-matching model for robot control.

- **Architecture**: {cfg.get('depth', '?')}-layer DiT, dim {cfg.get('model_dim', '?')}
- **Context frames**: {cfg.get('num_context_frames', '?')} @ {cfg.get('fps', '?')} fps
- **Future frames**: {cfg.get('num_future_frames', '?')}
- **Action dim**: {cfg.get('action_dim', '?')} @ {cfg.get('action_fps', '?')} fps
- **Cameras**: {', '.join(camera_names)}

## Usage

With [lerobot](https://github.com/huggingface/lerobot):

```bash
pip install "lerobot_policy_lewam @ git+https://github.com/ErykHalicki/lerobot_policy_lewam.git"
lerobot-record --policy.type lewam --policy.pretrained_path {repo_id}
```
"""
    (output_dir / "README.md").write_text(card)

    if repo_id:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            repo_type="model",
            commit_message="Upload LeWAM weights",
        )
        print(f"Pushed to https://huggingface.co/{repo_id}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LeWAM checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for HF format")
    parser.add_argument("--cameras", type=str, nargs="+", default=["image1", "image2"],
                        help="Camera names (default: image1 image2)")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID to push to")
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir, args.cameras, args.repo_id)
