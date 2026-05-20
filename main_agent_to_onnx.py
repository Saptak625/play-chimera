import torch
import os

from main_env import CinchMainEnv
from get_agent import get_agent
from obs_utils_torch import build_mlp_obs

MAIN_NET_CHECKPOINT_PATH = os.path.join("checkpoints_large_batches_2", "ckpt_5800.pt")
DEVICE = "cpu"

env = CinchMainEnv()
model = get_agent(env, MAIN_NET_CHECKPOINT_PATH)
legal = env.legal_actions()
obs = env._get_obs()
x = build_mlp_obs(obs).to(DEVICE)  # Add batch dimension and move to device

# 3. Export to an ONNX file
torch.onnx.export(
    model, 
    (x),
    "main_net.onnx",
    input_names=["observation", "legal_actions"],
    output_names=["action_logits", "state_value"],
    opset_version=18
)
print("Model exported to main_net.onnx")