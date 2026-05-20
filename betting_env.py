import numpy as np
import random
import torch
import uuid

from main_env import CinchMainEnv, create_deck, card_to_index, SUITS, RANKS

REWARDS_SCALE = 1
BETS = [0, 1, 2, 3, 4, 8]

class CinchBettingEnv:
    def __init__(self, dealer=3, seed=None, debug=False, no_reset=False):
        """
        Create a new betting environment.
        
        Args:
            dealer: The player who is the dealer.
            seed: The random seed for reproducibility.
            debug: Whether to print debug information.
            no_reset: Whether to reset the environment.
        """
        self.debug = debug
        self.game_uuid = uuid.uuid4()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if not no_reset:
            self.reset(dealer=dealer)

    def reset(self, dealer=3, no_reset=False):
        """
        Reset the environment to the initial state.
        """
        self.done = False
        self.main_game = None
        self.points_team = [0, 0]
        self.points_target = 15
        self.current_player = (dealer + 1) % 4
        self.dealer = dealer
        self.rounds_played = 0
        self.max_rounds = 100
        self.history = []
        if not no_reset:
            self.round_reset()
        return self._get_obs()

    def round_reset(self):
        """
        Reset the environment to the initial state.
        """
        self.betting_player = None
        self.bet = None
        self.trump = None
        self.got_bet = None
        self.winning_team = None
        self.deck = create_deck()
        random.shuffle(self.deck)
        self.starting_hands = [self.deck[i*9:(i+1)*9] for i in range(4)]
        self.bets = []
        self.trumps_declared = []
        self.rounds_played += 1
        return self._get_obs()
        
    def legal_actions(self):
        """
        Get the legal actions for the current player.
        """
        all_actions = BETS
        max_current_bet = max(self.bets) if self.bets else 0
        inds = [i for i, bet in enumerate(all_actions) if bet > max_current_bet]
        if not (self.current_player == self.dealer and max_current_bet == 0):
            # Index 0 is always legal.
            inds = [0] + inds
        # If the current player is the dealer, they can always overshoot.
        if self.current_player == self.dealer and 5 not in inds:
            inds = inds + [5]
        return inds

    def _get_obs(self):
        """
        Constructs the observation for the current player, including their hand, the trump suit, and the current trick.
        
        Returns:
            dict: A dictionary containing the player's hand as a binary vector, the trump suit as an index, and the current trick as a binary vector.
        """
        if self.debug:
            print(f'Player hands: {self.starting_hands[self.current_player]}')

        hand_vec = [0] * 52
        for c in self.starting_hands[self.current_player]:
            hand_vec[card_to_index(c)] = 1

        legal = self.legal_actions()

        which_team_bet = -1
        if self.bets:
            max_bet_idx = np.argmax(self.bets)
            which_team_bet = int((len(self.bets) - max_bet_idx - 1) % 2) # If 0 then opposite team is winning bet, if 1 then own team is winning bet.

        bets = [0, 0, 0, 0]
        for i, bet in enumerate(self.bets):
            bets[i] = bet

        obs = {
            "hand": hand_vec,
            "legal_actions": legal,
            "bets": bets,
            "player": self.current_player,
            "dealer": self.dealer,
            "bet_pos": len(self.bets),
            "which_team_bet": which_team_bet, # -1 if no bets, 0 if opposite team is winning bet, 1 if own team is winning bet.
            "team_points": self.points_team[self.current_player % 2],
            "opponent_points": self.points_team[(self.current_player + 1) % 2],
            "points_target": self.points_target,
            "rounds_played": self.rounds_played,
            "max_rounds": self.max_rounds
        }
        return obs
    
    def declare_trump_suit(self, trump):
        """
        Declare the trump suit for the current round. The betting player must have already been determined.

        Args:
            trump: The index of the trump suit (0-3 for the four suits).
        """
        self.trumps_declared[(self.betting_player - self.current_player) % 4] = trump

    def step(self, action):
        """
        Executes the given action (betting and declaring a trump suit) for the current player. 
        The trump suit is just kept in mind until all bets are declared.
        
        Args:
            action (bet, trump): A tuple containing the indices 0-5 indicating the bet (0, 1, 2, 3, 4, or 8) and the trump suit (0-3 for the four suits).
        Returns:
            tuple: A tuple containing the new observation, the rewards for all players, a boolean 
            indicating if the episode is done, and an empty info dictionary.
        """
        bet_idx, trump = action
        if len(self.bets) < 4:
            bet = BETS[bet_idx]
            if self.debug:
                print(f'Player {self.current_player} bets {bet} and wants to play trump {SUITS[trump] if trump is not None else "not declared"}\n')

            self.bets.append(bet)
            self.trumps_declared.append(trump)
            self.current_player = (self.current_player + 1) % 4
        self.rewards = [0, 0, 0, 0]

        # Check if all players have bet.
        if len(self.bets) == 4:
            # Declare the trump suit based on the max bet.
            max_bet = max(self.bets)
            inds = [i for i, b in enumerate(self.bets) if b == max_bet]
            max_bet_idx = inds[-1] # If two players have the same max bet, the dealer who bets last wins the tiebreaker.
            self.betting_player = (self.current_player + max_bet_idx) % 4 # Track the player who made the bet.
            self.bet = self.bets[max_bet_idx]
            if self.trumps_declared[max_bet_idx] is None:
                # Wait until the state gets updated with the declared trump suit before proceeding, since the main game needs to know the trump suit.
                return self._get_obs(), self.rewards, self.done, {}
            self.trump = SUITS[self.trumps_declared[max_bet_idx]]
            if self.debug:
                print(f'All players have bet. Player {self.betting_player} declared trump suit {self.trump} for bet {self.bet}. Starting main game.')
            
            # Reorder initial hands so that the betting player is first, and rotate the rest accordingly.
            initial_hands = self.starting_hands[self.betting_player:] + self.starting_hands[:self.betting_player]

            # Create a new main game environment based on the betting stage results if it doesn't already exist.
            if self.main_game is None:
                self.main_game = CinchMainEnv.from_betting_stage(initial_hands, self.trump, self.deck)
            if not self.main_game.done:
                return self._get_obs(), self.rewards, self.done, {}
            bet_points = self.main_game.bet_points
            self.main_game = None
            self.got_bet = bet_points[0] >= self.bet or (self.bet == 8 and bet_points[0] == 4)
            self.points_team[self.betting_player % 2] += max(bet_points[0], self.bet) if self.got_bet else -self.bet
            self.points_team[(self.betting_player + 1) % 2] += bet_points[1]
            self._round_rewards(bet_points)
            if self.debug:
                print(f'Main game finished. Points won by each team: {bet_points}')
                print(f'Points team 0: {self.points_team[0]}, Points team 1: {self.points_team[1]}, Winning team: {self.winning_team}')

            self.history.append({
                "betting_player": self.betting_player,
                "bets": self.bets.copy(),
                "trumps_declared": self.trumps_declared.copy(),
                "bet": self.bet,
                "trump": self.trump,
                "got_bet": self.got_bet,
                "bet_points": bet_points,
                "points_team": self.points_team.copy()
            })

            if max(self.points_team) >= self.points_target:
                if [i for i, points in enumerate(self.points_team) if points >= self.points_target] == [0, 1]:
                    # If both teams reach the target in the same round, the team that bet wins. 
                    # This can only happen if the betting team got their bet, since otherwise the opposing team would get points and the betting team would lose points.
                    self.winning_team = self.betting_player % 2
                else:
                    self.winning_team = 0 if self.points_team[0] >= self.points_target else 1
                self.done = True
                self._terminal_rewards()
                if self.debug:
                    print(f'Game finished. Final points: {self.points_team}. Rewards: {self._terminal_rewards()}')
            # elif self.rounds_played >= self.max_rounds:
            #     self.done = True
            #     # No terminal rewards if max rounds reached, just end the episode.
            #     # The agents already got feedback from the round rewards.
            #     # Even more reward if they get to 15 points and even more bonus reward for getting there in fewer rounds.
            #     if self.debug:
            #         print(f'Max rounds reached. Final points: {self.points_team}. Rewards: {self._terminal_rewards()}')
            else:
                # Reset for the next round.
                self.dealer = self.current_player # The next dealer is the player who started the previous round.
                self.current_player = (self.current_player + 1) % 4
                self.round_reset()

        return self._get_obs(), self.rewards, self.done, {}
    
    def _round_rewards(self, bet_points):
        """
        Calculate the rewards for each player based on the outcome of the main game and the bets.

        Args:
            bet_points: A list containing the points won by the betting team and the opposing team in the main game.
        """
        if self.debug:
            print(f'Calculating rewards. Bet points: {bet_points}, Bet: {self.bet}, Betting player: {self.betting_player}, Got bet: {self.got_bet}')
        rewards = [0, 0, 0, 0]
        if self.got_bet:
            # If the betting team got their bet, they get points equal to the bet, and the other team loses points equal to the bet.
            rewards[self.betting_player] = 0.01 * REWARDS_SCALE * bet_points[0] 
            rewards[(self.betting_player + 2) % 4] = 0.01 * REWARDS_SCALE * bet_points[0]
        else:
            # If the betting team did not get their bet, they lose points equal to the bet, and the other team gets points equal to the bet.
            rewards[self.betting_player] = -0.01 * REWARDS_SCALE * self.bet
            rewards[(self.betting_player + 2) % 4] = -0.01 * REWARDS_SCALE * self.bet

        # Give the other team a reward based on how many points they got in the main game, scaled by the bet.
        rewards[(self.betting_player + 1) % 4] = 0.01 * REWARDS_SCALE * bet_points[1]
        rewards[(self.betting_player + 3) % 4] = 0.01 * REWARDS_SCALE * bet_points[1]
        self.rewards = rewards

    def _terminal_rewards(self):
        """
        Calculate the terminal rewards for each player based on the final points of each team.
        """
        rewards = [0, 0, 0, 0]
        for i in range(4):
            if (i % 2) == self.winning_team:
                rewards[i] = 1 * REWARDS_SCALE + 0.01 * REWARDS_SCALE * (self.max_rounds - self.rounds_played)
            else:
                rewards[i] = -1 * REWARDS_SCALE
        self.rewards = rewards

    def to_dict(self):
        """
        Freeze the current state of the environment into a dictionary for logging or analysis.
        Should contain all relevant information about the current state of the environment.

        Returns:
            dict: A dictionary containing the current state of the environment.
        """
        # Make sure all properties of the environment are included in the dictionary.
        return {
            "game_uuid": str(self.game_uuid),
            "done": self.done,
            "main_game": self.main_game.to_dict() if self.main_game else None,
            "points_team": self.points_team.copy(),
            "points_target": self.points_target,
            "current_player": self.current_player,
            "dealer": self.dealer,
            "rounds_played": self.rounds_played,
            "max_rounds": self.max_rounds,
            "betting_player": self.betting_player,
            "bet": self.bet,
            "trump": self.trump,
            "got_bet": self.got_bet,
            "winning_team": self.winning_team,
            "deck": self.deck.copy(),
            "starting_hands": [hand.copy() for hand in self.starting_hands],
            "bets": self.bets.copy(),
            "trumps_declared": self.trumps_declared.copy(),
            "history": self.history.copy(),
        }
    
    def from_dict(state_dict):
        """
        Load the state of the environment from a dictionary.

        Args:
            state_dict: A dictionary containing the state of the environment.
        """
        env = CinchBettingEnv(no_reset=True)
        print('Loading environment from dict with state:', state_dict)  # Debugging line to check the state being loaded into the environment
        env.game_uuid = state_dict["game_uuid"]
        env.done = state_dict["done"]
        env.main_game = CinchMainEnv.from_dict(state_dict["main_game"]) if state_dict["main_game"] else None
        env.points_team = state_dict["points_team"].copy()
        env.points_target = state_dict["points_target"]
        env.current_player = state_dict["current_player"]
        env.dealer = state_dict["dealer"]
        env.rounds_played = state_dict["rounds_played"]
        env.max_rounds = state_dict["max_rounds"]
        env.betting_player = state_dict["betting_player"]
        env.bet = state_dict["bet"]
        env.trump = state_dict["trump"]
        env.got_bet = state_dict["got_bet"]
        env.winning_team = state_dict["winning_team"]
        env.deck = state_dict["deck"].copy()
        env.starting_hands = [hand.copy() for hand in state_dict["starting_hands"]]
        env.bets = state_dict["bets"].copy()
        env.trumps_declared = state_dict["trumps_declared"].copy()
        env.history = state_dict["history"].copy()
        return env

if __name__ == '__main__':
    import os
    env = CinchBettingEnv(seed=42, debug=True, no_reset=True)
    obs = env.reset()

    while not env.done:
        print(obs)
        legal_actions = env.legal_actions()
        bet = random.choice(legal_actions)
        trump = random.randint(0, 3)
        action = (bet, trump)
        obs, rewards, done, _ = env.step(action)
        print(f'Action taken: {action}, Rewards: {rewards}, Done: {done}')
        input('Press Enter to continue...')