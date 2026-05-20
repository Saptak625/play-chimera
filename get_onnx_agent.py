import os
import numpy as np
import onnxruntime as ort

from betting_env import BETS

DEVICE = "cpu"

class ONNXAgent:
    def __init__(self, session: ort.InferenceSession):
        self.session = session

    def predict(self, x: np.ndarray):
        """
        Run ONNX inference.
        """

        inputs = {
            self.session.get_inputs()[0].name:
                x.astype(np.float32)
        }

        outputs = self.session.run(None, inputs)

        return outputs


def get_agent(env, onnx_path: str):
    """
    Load ONNX Runtime session.
    """

    providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        onnx_path,
        providers=providers
    )

    model = ONNXAgent(session)
    return model


def select_action(model, obs, legal, obs_builder, post_process_bet_logits=False):
    x = obs_builder(obs)
    logits, _ = model.predict(x)

    # Mask out illegal actions by setting their logits to a very low value.
    masked = np.full_like(logits, -1e9)
    masked[legal] = logits[legal]

    # print("Model types:", type(model))
    # print("Legal actions:", legal)
    # print("Logits:", logits)
    # print("Masked logits:", masked)
    action = np.argmax(masked)
    # print("Choice:", action)

    if post_process_bet_logits:
        # print("Post-processing bet logits:", (int(action % len(BETS)), int(action // len(BETS))))
        return [int(action % len(BETS)), int(action // len(BETS))]

    return [int(action)]

if __name__ == "__main__":
    from main_env import CinchMainEnv, SUITS, RANKS
    from obs_utils import build_mlp_obs

    env = CinchMainEnv()

    onnx_path = os.path.join(
        "static",
        "main_net",
        "main_net.onnx"
    )

    model = get_agent(env, onnx_path)

    obs = env.reset()
    legal = env.legal_actions()

    action = select_action(
        model=model,
        obs=obs,
        legal=legal,
        obs_builder=build_mlp_obs
    )

    print("Action selected:", action)