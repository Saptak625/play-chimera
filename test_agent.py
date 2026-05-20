import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from obs_utils import build_betting_obs
from betting_env import CinchBettingEnv, BETS, SUITS
from get_agent import get_agent

DEVICE = "cpu"
BETTING_NET_CHECKPOINT_PATH = os.path.join("checkpoints_resnet-mlp-modified-agent", "ckpt_2400.pt")
MAIN_NET_CHECKPOINT_PATH = os.path.join("main_net", "checkpoints_large_batches_2", "ckpt_5800.pt")
DEBUG = True

def evaluate_model(model, games=500):
    """
    Evaluate the trained model against a random agent.
    
    Parameters:
    - model: The trained policy/value network to evaluate.
    - games: The number of games to play against the random agent for evaluation.
    Returns:
    - game_point_diff: A list of the point differences between the teams at the end of each game.
    - results: A list of bet points for each game, where 1 indicates a win for the trained model and 0 indicates a loss.
    - rounds: A list of the number of rounds played in each game.
    - histories: A list of the game histories for each game.
    """
    env = CinchBettingEnv(MAIN_NET_CHECKPOINT_PATH, debug=DEBUG)

    game_point_diff = []
    result = []
    rounds = []
    histories = []

    actions = [f'{b}{s}' for s in SUITS for b in BETS]

    for _ in tqdm(range(games), desc="Evaluating against random agent"):

        obs = env.reset()

        done = False

        while not done:

            current_player = env.current_player

            legal = env.legal_actions()

            x = build_betting_obs(obs).to(DEVICE)

            with torch.no_grad():
                logits, _ = model(x, [i * len(BETS) + l for l in legal for i in range(4)])

            if DEBUG:
                # Find the top 3 actions and their probabilities for debugging.
                probs = torch.softmax(logits, dim=-1)
                topk = min(3, len(legal))
                topk_probs, topk_indices = torch.topk(probs, k=topk)
                topk_actions = [actions[i] for i in topk_indices]
                print(f"Player {current_player} legal actions: {[actions[i] for i in legal]}")
                print(f"Player {current_player} top {topk} actions: {list(zip(topk_actions, topk_probs))}")
            action = torch.argmax(logits).item()

            obs, rewards, done, _ = env.step((action % len(BETS), action // len(BETS)))

        if max(env.points_team) >= 15:
            result.append(1 if env.points_team[0] > 15 else -1)
        else:
            result.append(0)
        game_point_diff.append(abs(env.points_team[0] - env.points_team[1]))
        rounds.append(env.rounds_played)
        histories.extend(env.history)
        if DEBUG:
            input("Press Enter to continue...")
            print("\n" + "="*100 + "\n")

    return game_point_diff, result, rounds, histories

if __name__ == "__main__":
    env = CinchBettingEnv(MAIN_NET_CHECKPOINT_PATH, debug=False)
    model = get_agent(env, BETTING_NET_CHECKPOINT_PATH)
    game_point_diff, results, rounds, histories = evaluate_model(model, games=500)
    print(f"Average win rate for the trained model: {sum(results)/len(results)*100:.2f}%")
    print(f"Number of wins for the trained model: {sum(abs(result) for result in results)}/{len(results)}")
    print(f"Average rounds per game: {sum(rounds)/len(rounds):.2f}")
    print(f"Number of games that ended without a team reaching 15 points: {results.count(0)}")
    print(f"Average point difference between the teams at the end of the game: {sum(game_point_diff)/len(game_point_diff):.2f}")
    print(f"Median point difference between the teams at the end of the game: {np.median(game_point_diff)}")

    # Make a histogram of the bets made by the agent.
    betting_player = []
    suits_bet = []
    winning_bet = []
    got_bet = []
    all_suits = []
    all_bets = []
    point_diff = []
    betting_diff = []
    bet_points_0 = []
    bet_points_1 = []
    for history in histories:
        betting_player.append(history["betting_player"])
        suits_bet.append(history["trump"])
        winning_bet.append(history["bet"])
        got_bet.append(history["got_bet"])
        all_suits.extend(history["trumps_declared"])
        all_bets.extend(history["bets"])

        bet_points = history["bet_points"]
        betting_team_points = max(bet_points[0], history["bet"]) if history["got_bet"] else -history["bet"]
        point_diff.append(betting_team_points - bet_points[1])
        betting_diff.append(betting_team_points - history["bet"])
        bet_points_0.append(betting_team_points)
        bet_points_1.append(bet_points[1])

    # Make a histogram of the game point differences at the end of the game.
    plt.figure()
    plt.hist(game_point_diff, bins=range(0, 25), align='left', edgecolor='black')
    plt.xlabel('Point Difference at End of Game')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Point Differences at the End of the Game')   
    plt.grid(axis='y')

    # Make a histogram of the betting players.
    plt.figure()
    plt.hist(betting_player, bins=range(5), align='left', edgecolor='black')
    plt.xlabel('Betting Player')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Betting Players')
    plt.xticks(range(4), ['Player 1', 'Player 2', 'Player 3', 'Player 4'])
    plt.grid(axis='y')

    # Make a histogram of the suits bet on by the agent.
    plt.figure()
    plt.hist(suits_bet, bins=range(5), align='left', edgecolor='black')
    plt.xlabel('Suit Bet On')
    plt.ylabel('Number of Bets')
    plt.title('Distribution of Suits Bet On by the Agent')
    plt.xticks(range(4), ['Hearts', 'Diamonds', 'Clubs', 'Spades'])
    plt.grid(axis='y')

    # Make a histogram of the suits declared by the agent.
    plt.figure()
    plt.hist(all_suits, bins=range(5), align='left', edgecolor='black')
    plt.xlabel('Suit Declared')
    plt.ylabel('Number of Declarations')
    plt.title('Distribution of Suits Declared by the Agent')
    plt.xticks(range(4), ['Hearts', 'Diamonds', 'Clubs', 'Spades'])
    plt.grid(axis='y')

    # Make a histogram of the bets made by the agent.
    plt.figure()
    plt.hist(all_bets, bins=range(9), align='left', edgecolor='black')
    plt.xlabel('Bet Made')
    plt.ylabel('Number of Bets')
    plt.title('Distribution of Bets Made by the Agent')
    plt.grid(axis='y')

    # Make a histogram of the winning bets made by the agent.
    plt.figure()
    plt.hist(winning_bet, bins=range(9), align='left', edgecolor='black')
    plt.xlabel('Winning Bet')
    plt.ylabel('Number of Bets')
    plt.title('Distribution of Winning Bets Made by the Agent')
    plt.grid(axis='y')

    # Make a histogram of the number of times the agent got their bet for a given bet.
    got_bet_counts = [0] * 6
    for bet, got in zip(winning_bet, got_bet):
        got_bet_counts[bet if bet != 8 else 5] += got

    # Normalize the got_bet_counts by the number of times each bet was made.
    bet_counts = [0] * 6
    for bet in winning_bet:
        bet_counts[bet if bet != 8 else 5] += 1
    got_bet_ratios = [got / count if count > 0 else 0 for got, count in zip(got_bet_counts, bet_counts)]

    plt.figure()
    plt.bar(range(6), got_bet_ratios, edgecolor='black')
    plt.xlabel('Winning Bet')
    plt.ylabel('Ratio of Times Got Bet Was Achieved')
    plt.title('Ratio of Times Got Bet Was Achieved for Each Winning Bet')
    plt.xticks(range(6), ['0', '1', '2', '3', '4', '8'])
    plt.grid(axis='y')

    # Make a histogram of the rounds played in each game.
    plt.figure()
    plt.hist(rounds, bins=range(1, max(rounds)+1), align='left', edgecolor='black')
    plt.xlabel('Rounds Played')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Rounds Played in Evaluation Games')
    plt.xticks(range(1, max(rounds)+1))
    plt.grid(axis='y')

    # Make a histogram of the point difference between the betting team and the opposing team at the end of the game.
    plt.figure()
    plt.hist(point_diff, bins=range(-15, 16), align='left', edgecolor='black')
    plt.axvline(x=np.mean(point_diff), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(point_diff):.2f}')
    plt.xlabel('Point Difference (Betting Team - Opposing Team)')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Point Difference at the End of the Game')
    plt.xticks(range(-15, 16))
    plt.grid(axis='y')

    # Make a histogram of the difference between the betting team points and the bet.
    plt.figure()
    plt.hist(betting_diff, bins=range(-15, 16), align='left', edgecolor='black')
    plt.axvline(x=np.mean(betting_diff), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(betting_diff):.2f}')
    plt.xlabel('Difference Between Betting Team Points and Bet')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Difference Between Betting Team Points and Bet') 
    plt.xticks(range(-15, 16))
    plt.grid(axis='y')

    # Put the histogram of the betting team points for each bet on the same plot.
    plt.figure()
    plt.hist([bet_points_0, bet_points_1], bins=range(-15, 16), align='left', edgecolor='black', label=['Betting Team Points', 'Opposing Team Points'])
    plt.axvline(x=np.mean(bet_points_0), color='blue', linestyle='dashed', linewidth=1, label=f'Betting Team Mean: {np.mean(bet_points_0):.2f}')
    plt.axvline(x=np.mean(bet_points_1), color='orange', linestyle='dashed', linewidth=1, label=f'Opposing Team Mean: {np.mean(bet_points_1):.2f}')
    plt.xlabel('Points')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Betting Team Points and Opposing Team Points')
    plt.xticks(range(-15, 16))
    plt.grid(axis='y')
    plt.legend()
    plt.show()