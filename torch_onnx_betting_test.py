from betting_env import CinchBettingEnv, BETS
from obs_utils import build_betting_obs as build_betting_obs_np

from get_agent import get_agent, select_action
from get_onnx_agent import get_agent as get_onnx_agent, select_action as select_onnx_action

import os

env = CinchBettingEnv()

onnx_path = os.path.join(
    "static",
    "betting_net",
    "betting_net.onnx"
)

pt_path = os.path.join(
    "checkpoints_resnet-mlp-modified-agent",
    "ckpt_4900.pt"
)

model = get_onnx_agent(env, onnx_path)
model2 = get_agent(env, pt_path)

obs = env.reset()
legal = env.legal_actions()

action = select_onnx_action(
    model=model,
    obs=obs,
    legal=[i * len(BETS) + l for i in range(4) for l in legal],
    obs_builder=build_betting_obs_np,
    post_process_bet_logits=True
)

print("Action selected by ONNX model:", action)

action = select_action(
    model=model2,
    checkpoint_path=pt_path,
    obs=obs,
    env=env
)
print("Action selected by PyTorch model:", action)