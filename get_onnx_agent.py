import os
import numpy as np
import onnxruntime as ort

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


def select_action(model, obs, env, obs_builder):
    legal = env.legal_actions()
    x = obs_builder(obs)
    logits, _ = model.predict(x)

    # Mask out illegal actions by setting their logits to a very low value.
    masked = np.full_like(logits, -1e9)
    masked[legal] = logits[legal]

    return int(np.argmax(masked))

if __name__ == "__main__":
    from main_env import CinchMainEnv, SUITS, RANKS
    from obs_utils import build_mlp_obs

    env = CinchMainEnv()

    onnx_path = os.path.join(
        "main_net",
        "main_net.onnx"
    )

    model = get_agent(env, onnx_path)

    obs = env.reset()

    action = select_action(
        model=model,
        obs=obs,
        env=env,
        obs_builder=build_mlp_obs
    )

    print("Action selected:", action)