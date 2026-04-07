# LeWAM Policy for LeRobot

A [LeRobot](https://github.com/huggingface/lerobot) policy plugin for [LeWAM](https://github.com/ErykHalicki/LeWAM), a flow-matching world-action model that jointly predicts future video embeddings and robot actions.

## Installation

```bash
uv pip install "lerobot_policy_lewam @ git+https://github.com/ErykHalicki/lerobot_policy_lewam.git"
```

This installs LeWAM and LeRobot as dependencies automatically. LeRobot auto-discovers the policy via the `lerobot_policy_*` package prefix.

## Usage

### Inference (real robot)

```bash
lerobot-record \
  --policy.type lewam \
  --policy.lewam_checkpoint_path /path/to/checkpoint.pt
```

### Training

```bash
lerobot-train \
  --policy.type lewam \
  --policy.lewam_checkpoint_path /path/to/checkpoint.pt \
  --dataset.repo_id your-hf-username/your-dataset
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lewam_checkpoint_path` | `None` | Path to a `.pt` checkpoint with baked config, weights, and norm stats |
| `num_ode_steps` | `10` | Euler integration steps during inference |
| `smooth_actions` | `True` | Apply Gaussian smoothing to predicted actions |
| `crop_size` | `256` | Input image crop size |
| `fps` | `5.0` | Video frame rate for context/future frames |
| `action_fps` | `30.0` | Action prediction frame rate |
| `num_context_frames` | `32` | Number of past frames for conditioning |
| `num_future_frames` | `8` | Number of future frames predicted (video embedding head) |
| `n_action_steps` | `48` | Number of action steps executed per chunk |
| `action_weight` | `1.0` | Loss weight for action prediction vs video embedding prediction |
| `lang_drop_rate` | `0.1` | Probability of dropping language tokens during training |


## Architecture

LeWAM uses a frozen VJEPA2 video encoder and frozen SmolVLM2 language encoder. Only the flow-matching transformer and action/video heads are trained. During inference, an ODE solver (Euler) iterates from noise to predicted video embeddings and action trajectories. The model operates entirely in VJEPA2's latent space, never decoding back to pixels.

Actions are predicted as relative velocities and converted to absolute joint positions via cumulative integration for robot control. The video embedding prediction acts as a regularizer during training.
