"""
Convert a LeWAM .pt checkpoint to HuggingFace format for use with lerobot.

This creates a directory with config.json + model.safetensors that can be
loaded via LeWAMPolicy.from_pretrained() or pushed to the Hub.

Usage:
    source .venv/bin/activate
    python -m lerobot_policy_lewam.export_to_hf \
        --checkpoint /path/to/checkpoint.pt \
        --output-dir /path/to/output \
        --repo-id ehalicki/lewam-so101  # optional, pushes to Hub
"""

import argparse
from pathlib import Path

def export(checkpoint_path: str, output_dir: str | Path, repo_id: str | None = None):
    from lewam.models.lewam import LeWAM

    print(f"Loading checkpoint from {checkpoint_path}...")
    model = LeWAM.from_checkpoint(checkpoint_path)

    output_dir = Path(output_dir)
    print(f"Saving HuggingFace format to {output_dir}...")
    model.save_pretrained(output_dir)

    cfg = model.config
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

## Usage

```python
from lewam.models.lewam import LeWAM
model = LeWAM.from_pretrained("{repo_id}")
```

Or with [lerobot](https://github.com/huggingface/lerobot):

```bash
pip install "lerobot_policy_lewam @ git+https://github.com/ErykHalicki/lerobot_policy_lewam.git"
lerobot-record --policy.type lewam --policy.pretrained_path {repo_id}
```
"""
    (Path(output_dir) / "README.md").write_text(card)

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
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repo ID to push to")
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir, args.repo_id)
