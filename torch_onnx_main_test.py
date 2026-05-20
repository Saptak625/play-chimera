from main_env import CinchMainEnv
from obs_utils import build_mlp_obs as build_mlp_obs_np

from get_agent import get_agent, select_action
from get_onnx_agent import get_agent as get_onnx_agent, select_action as select_onnx_action

import os

env = CinchMainEnv()

onnx_path = os.path.join(
    "static",
    "main_net",
    "main_net.onnx"
)

pt_path = os.path.join(
    "checkpoints_large_batches_2", 
    "ckpt_5800.pt"
)

model = get_onnx_agent(env, onnx_path)
model2 = get_agent(env, pt_path)

obs = env.reset()
legal = env.legal_actions()

action = select_onnx_action(
    model=model,
    obs=obs,
    legal=legal,
    obs_builder=build_mlp_obs_np
)

print("Action selected by ONNX model:", action)

action = select_action(
    model=model2,
    checkpoint_path=pt_path,
    obs=obs,
    env=env
)
print("Action selected by PyTorch model:", action)