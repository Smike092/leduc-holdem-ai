import abc

import numpy as np
from rlcard.envs import Env
from rlcard.games.leducholdem import game


class Agent(object):
    def __init__(self):
        super(Agent, self).__init__()

    @abc.abstractmethod
    def get_action(self, game_state):
        return

    def stop_running(self):
        pass


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, depth=2):
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class ExpectiminimaxAgent(MultiAgentSearchAgent):
    """
    Implement an Expectiminimax agent compatible with RLcard API
    """

    def __init__(self, env:Env, depth=2, heuristic='ratio_hands'):
        super(ExpectiminimaxAgent, self).__init__(depth)
        self.env = env
        self.use_raw = True
        self.heuristic = self.load(heuristic)

    def get_state(self, player_id):
        ''' Get state_str of the player
        Args:
            player_id (int): The player id
        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return list(state['legal_actions'].keys())

    def step(self, state):
        return self.get_action(state)

    def eval_step(self, state):
        return self.step(state), None

    def load(self, heuristic):
        if heuristic == 'ratio_hands':
            return self.ratio_hands

    def ratio_hands(self, player_id):
        """
        compare the number of hands that win vs lose, and average the payoff according
        """
        win, tie, lose = 0, 0, 0
        player0_card = self.env.game.players[player_id].hand
        public_card = self.env.game.public_card
        for player1_card in self.env.game.dealer_expecti.deck:
            if not public_card:
                for public_card in self.env.game.dealer_expecti.deck:
                    # check if one the cards is the same as the public card.
                    if player0_card == public_card:
                        win += 1
                    elif player1_card == public_card:
                        lose += 1
                    else:
                        if player0_card.rank > player1_card.rank:
                            win += 1
                        elif player0_card.rank > player1_card.rank:
                            lose += 1
                        else:
                            tie +=1
            else:
                if player0_card == public_card:
                    win += 1
                elif player1_card == public_card:
                    lose += 1
                else:
                    if player0_card.rank > player1_card.rank:
                        win += 1
                    elif player0_card.rank > player1_card.rank:
                        lose += 1
                    else:
                        tie += 1
        total = 0
        for p in self.env.game.players:
            total += p.in_chips
        ratio = (win-lose)/(win+lose+tie)
        return (ratio * total)/self.env.game.big_blind

    def get_action(self, state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def function_minimax(depth, player_id, action):
            over = self.env.game.round_over
            self.env.game.round_over = False
            if over:  # first round is over, luck agent play
                value = 0
                for card in self.env.game.dealer_expecti.deck:
                    self.env.game.public_card = card
                    value += function_minimax(depth, player_id, action)
                    self.env.step_back()
                value /= len(self.env.game.dealer_expecti.deck)
                self.env.step(action, raw_action=True)
            else:
                self.env.step(action, raw_action=True)
                if self.env.is_over():
                    return self.env.get_payoffs()[player_id] # return accurate payoff
                elif depth == 0:
                    return self.heuristic(player_id)# the heuristic function of the node
                current_player = self.env.get_player_id()
                legal_actions = self.env._get_legal_actions()
                if current_player == player_id:
                    value = -np.infty
                    for next_action in legal_actions:
                        value = max(value, function_minimax(depth - 1, player_id, next_action))
                        self.env.step_back()
                else:
                    value = np.inf
                    for next_action in legal_actions:
                        value = min(value, function_minimax(depth - 1, player_id, next_action))
                        self.env.step_back()
            return value

        self.env.game.planning = True
        current_player = self.env.get_player_id()
        legal_actions = self.env._get_legal_actions()
        lst = []
        for i, action in enumerate(legal_actions):
            lst.append((function_minimax(self.depth, current_player, action), i, action))
            self.env.step_back()
        best = max(lst)
        self.env.game.planning = False
        return best[2]

