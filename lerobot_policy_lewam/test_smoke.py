"""Smoke test: random-init LeWAM policy on a streaming dataset sample."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot_policy_lewam import LeWAMConfig, LeWAMPolicy

REPO_ID = "aivanni/so101-puzzle-v9"

config = LeWAMConfig(
    input_features={
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    crop_size=256,
)

print("Building policy (random init)...")
policy = LeWAMPolicy(config)
policy.eval()

print("Loading streaming dataset...")
ds = StreamingLeRobotDataset(REPO_ID, episodes=[0], shuffle=False)
sample = next(iter(ds))
print(f"Task: {sample['task']}")

batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
batch["task"] = [sample["task"]]

print("Running select_action (7 steps)...")
actions = []
with torch.no_grad():
    for step in range(7):
        action = policy.select_action(batch)
        actions.append(action.squeeze().cpu())
        print(f"  step {step}: {action.squeeze()[:3].tolist()}")

actions = torch.stack(actions)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

front_img = sample["observation.images.front"].permute(1, 2, 0).clamp(0, 1).numpy()
axes[0].imshow(front_img)
axes[0].set_title("GT: front")
axes[0].axis("off")

wrist_img = sample["observation.images.wrist"].permute(1, 2, 0).clamp(0, 1).numpy()
axes[1].imshow(wrist_img)
axes[1].set_title("GT: wrist")
axes[1].axis("off")

joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
for i in range(actions.shape[1]):
    axes[2].plot(actions[:, i].numpy(), label=joint_names[i])
axes[2].set_title("Predicted actions (random init)")
axes[2].set_xlabel("step")
axes[2].set_ylabel("position")
axes[2].legend(fontsize=7)

plt.suptitle(f"Task: {sample['task']}")
plt.tight_layout()
plt.savefig("test_policy_output.png", dpi=150)
print("Saved test_policy_output.png")
