from flask import Flask, render_template, flash, request, abort
from flask_assets import Environment
from flask_cors import CORS
from assets import bundles
from functools import wraps
from urllib.parse import urlparse
import json
import base64
import os
import torch

from betting_env import CinchBettingEnv, BETS
from main_env import CinchMainEnv
from get_agent import get_agent
from obs_utils import build_mlp_obs, build_betting_obs

DEVICE = "cpu"
BETTING_NET_CHECKPOINT_PATH = os.path.join("betting_net", "checkpoints_resnet-mlp-modified-agent", "ckpt_4900.pt")
MAIN_NET_CHECKPOINT_PATH = os.path.join("main_net", "checkpoints_large_batches_2", "ckpt_5800.pt")

app = Flask(__name__)
app.config['SECRET_KEY'] = '9efb7fe9af62768bdc7adc9200cf1159'
CORS(app)

assets = Environment(app)
assets.register(bundles)

# Custom decorator to enforce the referrer
def require_play_referrer(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        referrer = request.headers.get('Referer')
        
        if referrer:
            # Parse the URL to safely check the path, ignoring domain/ports
            path = urlparse(referrer).path
            if path == '/play':
                return f(*args, **kwargs)
                
        # If no referrer or incorrect referrer, block the request
        abort(403, description="Access denied: This API can only be accessed from the play page.")
    return decorated_function

@app.after_request
def add_security_headers(response):
    # Allows external web apps like Netron to safely fetch data from your localhost
    response.headers["Access-Control-Allow-Origin"] = "https://netron.app"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    
    # CRITICAL: This bypasses the Private Network Access restriction on localhost
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    
    return response

# ===============================================================
# GUI ENDPOINTS
# ===============================================================
@app.route('/')
def index():
    return render_template('index.html', data=None)


@app.route('/play', methods=['GET'])
def play():
    return render_template('play.html', data=None)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', data=None)


# ===============================================================
# API ENDPOINTS
# ===============================================================
@app.route('/api/start-game', methods=['POST'])
@require_play_referrer
def start_game():
    decoded_data = base64.b64decode(request.data).decode()
    data = json.loads(decoded_data)
    dealer = data.get('dealer', 3)
    env = CinchBettingEnv(dealer=dealer)
    json_response = env.to_dict()
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/get-obs', methods=['POST'])
@require_play_referrer
def get_obs():
    decoded_state = base64.b64decode(request.data).decode()
    state = json.loads(decoded_state)
    if state.get('main_game') is None:
        env = CinchBettingEnv.from_dict(state_dict=state)
        obs = env._get_obs()
    else:
        env = CinchMainEnv.from_dict(state_dict=state['main_game'])
        obs = env._get_obs()
    data = json.dumps(obs)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/play-action', methods=['POST'])
@require_play_referrer
def play_action():
    decoded_data = base64.b64decode(request.data).decode()
    state = json.loads(decoded_data)
    env = CinchBettingEnv.from_dict(state_dict=state)
    main_env = None
    if state.get('main_game') is None or state['main_game']['done']:
        decision = state.get('decision')
        obs, reward, done, info = env.step(decision)
    else:
        main_env = CinchMainEnv.from_dict(state_dict=state['main_game'])
        decision = state['main_game'].get('decision')
        obs, reward, done, info = main_env.step(decision[0]) # decision is expected to be a list like [action_index]
    json_response = env.to_dict()
    if main_env is not None:
        json_response['main_game'] = main_env.to_dict()  # Include the updated main game state in the response if it exists
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/chimera-action', methods=['POST'])
@require_play_referrer
def chimera_action():
    decoded_data = base64.b64decode(request.data).decode()
    state = json.loads(decoded_data)
    if state.get('main_game') is None:
        env = CinchBettingEnv()
        model = get_agent(env, BETTING_NET_CHECKPOINT_PATH)
        env = CinchBettingEnv.from_dict(state_dict=state)
        legal = env.legal_actions()
        obs = env._get_obs()
        x = build_betting_obs(obs).to(DEVICE)

        with torch.no_grad():
            logits, _ = model(x, [i * len(BETS) + l for l in legal for i in range(4)])

        action = torch.argmax(logits).item()
        data = json.dumps([action % len(BETS), action // len(BETS)])  # Convert back to (bet, suit) format
    else:
        env = CinchMainEnv()
        model = get_agent(env, MAIN_NET_CHECKPOINT_PATH)
        env = CinchMainEnv.from_dict(state_dict=state['main_game'])
        legal = env.legal_actions()
        obs = env._get_obs()
        x = build_mlp_obs(obs).to(DEVICE)  # Add batch dimension and move to device

        with torch.no_grad():
            logits, _ = model(x, legal)

        data = json.dumps([torch.argmax(logits).item()])
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/declare-trump', methods=['POST'])
@require_play_referrer
def declare_trump():
    decoded_data = base64.b64decode(request.data).decode()
    state = json.loads(decoded_data)
    trump_suit = state.get('declared_trump')
    env = CinchBettingEnv.from_dict(state_dict=state)
    env.declare_trump_suit(trump_suit)
    env.step((None, None))  # Take a dummy step to advance the game state after declaring trump
    json_response = env.to_dict()
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
