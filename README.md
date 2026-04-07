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

Run the policy on a powerful machine (e.g. a desktop GPU) and execute actions on a separate machine connected to the robot. The two communicate over gRPC.

**1. Start the policy server (PC with GPU)**

```bash
python -m lerobot_policy_lewam.serve_lewam \
  --host=0.0.0.0 \
  --port=8080
```

The server waits for the robot client to connect. When the client connects, it tells the server which model to load. The server downloads the model from HuggingFace Hub (or loads from a local path) and begins accepting observations.

**2. Start the robot client (laptop with robot)**

```bash
python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem58760431541 \
  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --robot.id=black \
  --task="pick up the puzzle piece" \
  --server_address=<PC_IP>:8080 \
  --policy_type=lewam \
  --pretrained_name_or_path=your-hf-username/lewam-model \
  --policy_device=cuda \
  --client_device=cpu
```

The client streams camera frames and robot state to the server, receives predicted action chunks back, and executes them on the robot.

To use a local model directory instead of HuggingFace Hub, pass a path:

```bash
--pretrained_name_or_path=/path/to/lewam-hf
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_ode_steps` | `10` | Euler integration steps during inference |
| `smooth_actions` | `True` | Apply Savitzky-Golay smoothing to predicted actions |
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
