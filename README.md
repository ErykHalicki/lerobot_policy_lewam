# LeWAM Policy for LeRobot

A [LeRobot](https://github.com/huggingface/lerobot) policy plugin for [LeWAM](https://github.com/ErykHalicki/LeWAM), a flow-matching world-action model that jointly predicts future video embeddings and robot actions.

## Installation

```bash
uv pip install "lerobot_policy_lewam @ git+https://github.com/ErykHalicki/lerobot_policy_lewam.git"
```

This installs LeWAM and LeRobot as dependencies automatically. LeRobot auto-discovers the policy via the `lerobot_policy_*` package prefix.

## Exporting a Checkpoint

LeWAM checkpoints must be converted to HuggingFace format before use with LeRobot. This produces a directory with `config.json` and `model.safetensors`.

```bash
python -m lerobot_policy_lewam.export_to_hf \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir ./lewam-hf
```

To also push to the HuggingFace Hub:

```bash
python -m lerobot_policy_lewam.export_to_hf \
  --checkpoint /path/to/checkpoint.pt \
  --output-dir ./lewam-hf \
  --repo-id your-hf-username/lewam-model
```

## Usage

### Inference (real robot)

```bash
lerobot-record \
  --policy.type lewam \
  --policy.pretrained_path your-hf-username/lewam-model
```

### Training

```bash
lerobot-train \
  --policy.type lewam \
  --policy.pretrained_path your-hf-username/lewam-model \
  --dataset.repo_id your-hf-username/your-dataset
```

### Remote Inference (policy server)

Run the policy on a GPU machine and execute actions on a separate machine connected to the robot. The client sends all context frames in a single batch, the server runs inference and returns an action chunk.

**1. Start the policy server (PC with GPU)**

```bash
python -m lerobot_policy_lewam.serve_lewam \
  --model ehalicki/lewam-so101-multitask-finetuned \
  --device cuda \
  --port 8080
```

**2. Start the robot client (laptop with robot)**

```bash
python src/lewam/scripts/rollout.py \
  --server <PC_IP> \
  --port 8080 \
  --robot-port /dev/tty.usbmodem5B141136531
```

The client maintains a 32-frame context buffer locally, JPEG-encodes and sends the full context with robot state and task to the server each inference cycle. The server returns predicted actions and PCA-projected future video embeddings for visualization. Actions are executed at 30fps with live rerun visualization of camera feeds and predicted futures.

To use a different model, pass `--model` with a HuggingFace repo ID or local path.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_ode_steps` | `2` | Euler integration steps during inference |
| `smooth_actions` | `True` | Apply Savitzky-Golay smoothing to predicted actions |
| `crop_size` | `224` | Input image crop size |
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
