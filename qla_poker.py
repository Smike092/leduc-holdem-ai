import random

import util


class PokerState():
    def __init__(self, state, action):
        self.state = state
        self.action = action

    def tostring(self):
        return self.state['raw_obs']['hand'] + " " + str(self.state['raw_obs']['public_card']) + " " + \
               str(self.state['action_record']) + " " + str(self.action)

    def __hash__(self):
        obs = (self.state['obs'] + [self.action]).tobytes()
        return hash(obs)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class QLA_Poker(object):
    """
    Implement a Q-Learning agent compatible with RLcard API
    """
    def __init__(self, epsilon=0.5, alpha=0.5, gamma=1):
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.Q_value = util.Counter()
        self.use_raw = False

    def getQValue(self, state, action):
        """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
        key = PokerState(state, action)
        return self.Q_value[key.tostring()]

    def getValue(self, state):
        """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
        if not state['legal_actions']:
            return 0.0
        state_q_values = []
        for action in state['legal_actions']:
            state_q_values.append(self.getQValue(state, action))
        return max(state_q_values)

    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        value_max = self.getValue(state)
        actions = state['legal_actions']
        best_actions = []
        for action in actions:
            if value_max == self.getQValue(state, action):
                best_actions.append(action)
        if best_actions:
            return random.choice(best_actions)
        return None

    def getAction(self, state):
        """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
        # Pick Action
        legalActions = state['legal_actions']
        if legalActions:
            if util.flipCoin(self.epsilon):
                actions = list(legalActions.keys())
                return random.choice(actions)
            else:
                return self.getPolicy(state)
        return None

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
        q_val = self.getQValue(state, action)
        val_next = self.getValue(nextState)
        key = PokerState(state, action)
        self.Q_value[key.tostring()] += self.alpha * (reward + self.discount * val_next - q_val)

    def step(self, state):
        return self.getAction(state)

    def eval_step(self, state):
        return self.getPolicy(state), None

    def feed(self, ts):
        state, action, reward, next_state, done = tuple(ts)
        state["action_record"] = state["action_record"][:-2]
        self.update(state, action, next_state, reward)
