import numpy as np
import rlcard
from rlcard.agents import RandomAgent
from rlcard.models.leducholdem_rule_models import LeducHoldemRuleAgentV1, LeducHoldemRuleAgentV2
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)
import rlcard.games.leducholdem.game

from expectiminimax import ExpectiminimaxAgent

MAX_DEPTH = 4


def evaluate_expectiminimax(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed, 'allow_step_back': True})

    # Init agents for tests
    test_agents = [RandomAgent(num_actions=env.num_actions), LeducHoldemRuleAgentV1(), LeducHoldemRuleAgentV2()]

    result_arr = np.zeros((MAX_DEPTH, len(test_agents)))

    for i in range(MAX_DEPTH):
        # Load models
        agent = ExpectiminimaxAgent(env=env, depth=i)
        for j in range(len(test_agents)):
            agents = [agent, test_agents[j]]
            env.set_agents(agents)

            # Evaluate
            result_arr[i][j] = tournament(env, args.num_games)[0]

    import plotly.graph_objects as go
    depths = ["Depth " + str(i) for i in range(MAX_DEPTH)]

    fig = go.Figure(data=[
        go.Bar(name='Random', x=depths, y=result_arr[:, 0]),
        go.Bar(name='Aggressive', x=depths, y=result_arr[:, 1]),
        go.Bar(name='Cautious', x=depths, y=result_arr[:, 2])
    ])
    # Change the bar mode
    fig.update_layout(title_text="Average Reward of Expectiminimax on " + str(args.num_games) + " steps", barmode='group')
    fig.write_image("experiments/expectiminimax.png")
    print("plot successfully saved")
    fig.show()