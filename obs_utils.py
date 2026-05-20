import numpy as np
from main_env import SUITS, RANKS

# =========================================================
# BETTING OBS (ONNX SAFE)
# =========================================================
def build_betting_obs(obs):
    """
    Convert the observation dictionary to a numpy array.
    The observation dictionary contains the following keys:
     - "hand": a binary vector of length 52 indicating which cards are in the player's hand
     - "legal_actions": a list of legal action indices for possible bets (0-5)
     - "bets": a float vector of length 4 indicating the current bets of each player, normalized by 8 (the maximum bet).
     - "player": a normalized integer from 0 to 3 indicating the player's position (0=first player, 1=second player, 2=third player, 3=fourth player)
     - "dealer": a normalized integer from 0 to 3 indicating the dealer's position (0=first player, 1=second player, 2=third player, 3=fourth player)
     - "bet_pos": a normalized integer from 0 to 3 indicating the player's position in the betting order (0=first to bet, 1=second to bet, etc.)
     - "which_team_bet": a binary vector of length 3 indicating the current state of the bets (0=opposite team is winning bet, 1=own team is winning bet, 2=no bets yet)
     - "team_points": a normalized float indicating the current points of the player's team, normalized by 15.
     - "opponent_points": a normalized float indicating the current points of the opposing team, normalized by 15.
     - "rounds_played": a normalized integer indicating how many rounds have been played, normalized by 30 (the maximum number of rounds in a game).
    """
    legal_vec = np.zeros(6, dtype=np.float32)
    legal_vec[obs["legal_actions"]] = 1.0

    which_team_bet_vec = np.zeros(3, dtype=np.float32)
    if obs["which_team_bet"] == -1:
        which_team_bet_vec[2] = 1.0
    else:
        which_team_bet_vec[obs["which_team_bet"]] = 1.0

    final = np.concatenate([
        obs["hand"].astype(np.float32),
        legal_vec,
        np.array(obs["bets"], dtype=np.float32) / 8.0,
        np.eye(4, dtype=np.float32)[obs["player"]] / 3.0,
        np.eye(4, dtype=np.float32)[obs["dealer"]] / 3.0,
        np.array([obs["bet_pos"]], dtype=np.float32) / 3.0,
        which_team_bet_vec,
        np.array([obs["team_points"]], dtype=np.float32) / obs["points_target"],
        np.array([obs["opponent_points"]], dtype=np.float32) / obs["points_target"],
        np.array([obs["rounds_played"]], dtype=np.float32) / obs["max_rounds"],
    ], dtype=np.float32)

    return final


# =========================================================
# MLP OBS (ONNX SAFE)
# =========================================================
def build_mlp_obs(obs):
    """
    Convert the observation dictionary to a numpy array.
    The observation dictionary contains the following keys:
     - "hand": a binary vector of length 52 indicating which cards are in the player's hand
     - "trick": a normalized integer vector of length 52 indicating which cards have been played in the current trick
     - "cards_played": a float vector (normalized by 1/24) of length 52 indicating which cards have been played in the game
     - "trump": a normalized integer from 0 to 3 indicating the trump suit (0=hearts, 1=diamonds, 2=clubs, 3=spades)
     - "player": a normalized integer from 0 to 3 indicating the player's position (0=first player, 1=second player, 2=third player, 3=fourth player)
     - "void": a flattened binary vector of length 16 indicating which suits the various players are void in based on the cards that have been played (4 suits x 4 players)
     - "trick_pos": a normalized integer from 0 to 3 indicating the player's position in the current trick (0=first to play, 1=second to play, etc.)
     - "lead_suit": a binary vector of length 5 indicating the lead suit of the current trick (one-hot encoding with an extra index for no lead suit)
     - "trick_winner": A binary vector of length 3 indicating the winner of the current trick (0=opposing team, 1=current team, 2=neither).
     - "trick_value": A normalized float indicating the value of the current trick based on the cards played so far. (Indicates whether the trick is currently a junk trick or a high value trick)
     - "can_win_mask": A binary vector of length 52 indicating which legal cards in the player's hand can currently win the trick based on the cards that have been played so far.
     - "trumps_at_start": a normalized integer list of length 4 indicating how many trump cards each player had before drawing from the deck
     - "count_from_deck": a normalized integer list of length 4 indicating how many cards each player has drawn from the deck
     - "count_from_dead_wood": a normalized integer list of length 4 indicating how many cards each player has drawn from the dead wood
     - "count_in_widow": an normalized integer indicating how many cards are currently in the widow (partially important in case certain point trump cards are missing from the game)
    """
    lead_vec = np.zeros(5, dtype=np.float32)
    if obs["lead_suit"] != -1:
        lead_vec[obs["lead_suit"]] = 1.0
    else:
        lead_vec[4] = 1.0

    trick_winner_vec = np.zeros(3, dtype=np.float32)
    if obs["trick_winner"] != -1:
        trick_winner_vec[obs["trick_winner"]] = 1.0
    else:
        trick_winner_vec[2] = 1.0

    final = np.concatenate([
        np.array(obs["hand"], dtype=np.float32),
        np.array(obs["trick"], dtype=np.float32) / 3.0,
        np.array(obs["cards_played"], dtype=np.float32) / 24.0,
        np.eye(4, dtype=np.float32)[obs["trump"]] / 3.0,
        np.eye(4, dtype=np.float32)[obs["player"]] / 3.0,
        np.array(obs["void"], dtype=np.float32),
        np.array([obs["trick_pos"]], dtype=np.float32) / 3.0,
        lead_vec,
        trick_winner_vec,
        np.array([obs["trick_value"]], dtype=np.float32),
        np.array(obs["can_win_mask"], dtype=np.float32),
        np.array(obs["trumps_at_start"], dtype=np.float32) / 6.0,
        np.array(obs["count_from_deck"], dtype=np.float32) / 6.0,
        np.array(obs["count_from_dead_wood"], dtype=np.float32) / 6.0,
        np.array([obs["count_in_widow"]], dtype=np.float32) / 5.0,
    ], dtype=np.float32)

    return final