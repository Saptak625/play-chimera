from flask import Flask, render_template, request, abort
# from flask_assets import Environment
# from assets import bundles
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from functools import wraps
from urllib.parse import urlparse
import json
import base64
import os
from dotenv import load_dotenv

from betting_env import CinchBettingEnv, BETS
from main_env import CinchMainEnv
from get_onnx_agent import get_agent as get_onnx_agent, select_action as select_onnx_action
from obs_utils import build_betting_obs as build_betting_obs_np, build_mlp_obs as build_mlp_obs_np

BETTING_NET_CHECKPOINT_PATH = os.path.join("static", "betting_net", "betting_net.onnx")
MAIN_NET_CHECKPOINT_PATH = os.path.join("static", "main_net", "main_net.onnx")

app = Flask(__name__)
app.config['SECRET_KEY'] = '9efb7fe9af62768bdc7adc9200cf1159'

# assets = Environment(app)
# assets.register(bundles)

# ===============================================================
# MONGODB SETUP
# ===============================================================
load_dotenv()  # Load environment variables from .env file
uri = os.getenv("MONGO_URI")
if not uri:
    raise ValueError("MONGO_URI environment variable not set. Please set it in your .env file.")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Try putting a betting env game state into the database to test the connection
db = client["cinch_game_db"]
games_collection = db["game_states"]

def save_to_mongodb(game_state):
    game_uuid = game_state["game_uuid"]
    save = False
    # Make a copy of the game state to avoid modifying the original object
    game_state_copy = {}
    copy_keys = ['done', 'main_game', 'points_team', 'current_player', 'dealer', 'rounds_played', 'deck', 'bets', 'trumps_declared']
    for key in copy_keys:
        if key in game_state:
            game_state_copy[key] = game_state[key]

    if game_state_copy.get('main_game') is not None:
        if game_state_copy['main_game'].get('card_num') == 0:  
            # Only save the main game state at the start of a new round, to avoid saving redundant states after every action in the main game
            save = True
            del game_state_copy['main_game'] # Remove the main game state.
        elif game_state_copy['main_game'].get('done', False):  # If the main game is done, we want to save the final state of the main game.
            save = True
            game_state_copy = game_state_copy['main_game']
            copy_keys = ['trump', 'trumps_at_start', 'count_in_widow', 'cards_played', 'done', 'bet_points']
            for key in copy_keys:
                if key in game_state:
                    game_state_copy[key] = game_state['main_game'][key]
    if save:
        games_collection.update_one(
            {"game_uuid": game_uuid},
            {"$push": {"states": game_state_copy}},
            upsert=True  # Create a new document if one with the same game_uuid doesn't exist
        )

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
    return response

# ===============================================================
# GUI ENDPOINTS
# ===============================================================
@app.route('/')
def index():
    return render_template('index.html', data=None)

@app.route('/how-to-play', methods=['GET'])
def how_to_play():
    return render_template('how_to_play.html', data=None)

@app.route('/play', methods=['GET'])
def play():
    return render_template('play.html', data=None)

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html', data=None)

@app.route('/api', methods=['GET'])
def api():
    return render_template('api.html', data=None)

# ===============================================================
# API ENDPOINTS
# ===============================================================
@app.route('/api/start-game', methods=['POST'])
# @require_play_referrer
def start_game():
    decoded_data = base64.b64decode(request.data).decode()
    data = json.loads(decoded_data)
    dealer = data.get('dealer', 3)
    env = CinchBettingEnv(dealer=dealer)
    json_response = env.to_dict()
    # Store the game state in MongoDB
    save_to_mongodb(json_response)
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/get-obs', methods=['POST'])
# @require_play_referrer
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
# @require_play_referrer
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
    # Store the game state in MongoDB
    save_to_mongodb(json_response)
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/chimera-action', methods=['POST'])
# @require_play_referrer
def chimera_action():
    decoded_data = base64.b64decode(request.data).decode()
    state = json.loads(decoded_data)
    if state.get('main_game') is None:
        env = CinchBettingEnv()
        model = get_onnx_agent(env, BETTING_NET_CHECKPOINT_PATH)
        env = CinchBettingEnv.from_dict(state_dict=state)
        legal = env.legal_actions()
        action = select_onnx_action(
            model=model,
            obs=env._get_obs(),
            legal=[i * len(BETS) + l for i in range(4) for l in legal],
            obs_builder=build_betting_obs_np,
            post_process_bet_logits=True
        )
    else:
        env = CinchMainEnv()
        model = get_onnx_agent(env, MAIN_NET_CHECKPOINT_PATH)
        env = CinchMainEnv.from_dict(state_dict=state['main_game'])
        legal = env.legal_actions()
        action = select_onnx_action(
            model=model,
            obs=env._get_obs(),
            legal=legal,
            obs_builder=build_mlp_obs_np
        )
    data = json.dumps(action)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

@app.route('/api/declare-trump', methods=['POST'])
# @require_play_referrer
def declare_trump():
    decoded_data = base64.b64decode(request.data).decode()
    state = json.loads(decoded_data)
    trump_suit = state.get('declared_trump')
    env = CinchBettingEnv.from_dict(state_dict=state)
    env.declare_trump_suit(trump_suit)
    env.step((None, None))  # Take a dummy step to advance the game state after declaring trump
    json_response = env.to_dict()
    # Store the game state in MongoDB
    save_to_mongodb(json_response)
    data = json.dumps(json_response)
    encoded_data = base64.b64encode(data.encode()).decode()
    return encoded_data

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
