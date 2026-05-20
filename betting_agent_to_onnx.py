import torch
import os

from betting_env import CinchBettingEnv, BETS
from get_agent import get_agent
from obs_utils import build_betting_obs

BETTING_NET_CHECKPOINT_PATH = os.path.join("betting_net", "checkpoints_resnet-mlp-modified-agent", "ckpt_4900.pt")
DEVICE = "cpu"

env = CinchBettingEnv()
model = get_agent(env, BETTING_NET_CHECKPOINT_PATH)
legal = env.legal_actions()
obs = env._get_obs()
x = build_betting_obs(obs).to(DEVICE)

# with torch.no_grad():
#     logits, _ = model(x, [i * len(BETS) + l for l in legal for i in range(4)])

# 3. Export to an ONNX file
torch.onnx.export(
    model, 
    (x, [i * len(BETS) + l for l in legal for i in range(4)]),
    "betting_net.onnx",
    input_names=["observation", "legal_actions"],
    output_names=["action_logits", "state_value"],
    opset_version=18
)
print("Model exported to betting_net.onnx")