import argparse

import rlcard
from rlcard.agents import RandomAgent, CFRAgent
from rlcard.models.leducholdem_rule_models import LeducHoldemRuleAgentV1, LeducHoldemRuleAgentV2
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger
)
from qla_poker import QLA_Poker
import numpy as np
import os
import matplotlib.pyplot as plt

EPSILON_R = 0.8
ALPHA_R = 0.001
GAMMA_R = 0.3

EPSILON_A = 0.8
ALPHA_A = 0.001
GAMMA_A = 0.5

EPSILON_C = 0.9
ALPHA_C = 0.01
GAMMA_C = 0.3


def train_cfr(args):
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make environments, CFR only supports Leduc Holdem
    env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )
    # Initialize the agent
    agent = CFRAgent(
        env,
        os.path.join(
            args.log_dir,
            'cfr_model',
        )
    )

    # Init agents for tests
    test_agents = [RandomAgent(num_actions=eval_env.num_actions), LeducHoldemRuleAgentV1(), LeducHoldemRuleAgentV2()]

    # Start training

    with Logger(args.log_dir) as logger:
        result_lst = np.zeros((len(test_agents), 2, args.num_episodes // args.evaluate_every))
        for episode in range(args.num_episodes):
            agent.train()
            print('\rIteration {}'.format(episode), end='')

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                agent.save()
                for i, test_agent in enumerate(test_agents):
                    agents = [agent, test_agent]
                    eval_env.set_agents(agents)
                    result_lst[i][0][episode // args.evaluate_every] = episode
                    result_lst[i][1][episode // args.evaluate_every] = tournament(
                        eval_env,
                        args.num_games,
                    )[0]
        fig, ax = plt.subplots()
        for i, name in enumerate(["random", "aggressive", "cautious"]):
            ax.plot(result_lst[i][0], result_lst[i][1], label=name)
        ax.set(xlabel='num_episodes', ylabel='reward')
        ax.legend()
        ax.grid()
        # save plot
        save_dir = os.path.dirname(logger.fig_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(logger.fig_path)


def train_qla(args):
    """
    Train QLA pendant 100000 iteration contre random utilisant les meilleurs hyperparametres.
    :param args:
    :return:
    """
    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
        config={
            'seed': 0,
        }
    )
    # Initialize the agent and use random agents as opponents
    qla_agents = [QLA_Poker(epsilon=EPSILON_R, alpha=ALPHA_R, gamma=GAMMA_R),
                  QLA_Poker(epsilon=EPSILON_A, alpha=ALPHA_A, gamma=GAMMA_A),
                  QLA_Poker(epsilon=EPSILON_C, alpha=ALPHA_C, gamma=GAMMA_C)]

    # train_agent = RandomAgent(num_actions=env.num_actions)
    # env.set_agents([agent, train_agent])

    # Init agents for testing
    test_agents = [RandomAgent(num_actions=eval_env.num_actions), LeducHoldemRuleAgentV1(), LeducHoldemRuleAgentV2()]

    # Start training
    with Logger(args.log_dir) as logger:
        result_lst = np.zeros((len(test_agents), 2, args.num_episodes // args.evaluate_every))
        for episode in range(args.num_episodes):
            for i in range(3):
                env.set_agents([qla_agents[i], test_agents[i]])
                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[0]:
                    qla_agents[i].feed(ts)

                # Evaluate the performance. Play with random agents.
                if episode % args.evaluate_every == 0:
                    agents = [qla_agents[i], test_agents[i]]
                    eval_env.set_agents(agents)
                    result_lst[i][0][episode // args.evaluate_every] = episode
                    result_lst[i][1][episode // args.evaluate_every] = tournament(
                        eval_env,
                        args.num_games,
                    )[0]
    fig, ax = plt.subplots()
    for i, name in enumerate(["random", "aggressive", "cautious"]):
        ax.plot(result_lst[i][0], result_lst[i][1], label=name)
    ax.set(xlabel='num_episodes', ylabel='reward')
    ax.legend()
    ax.grid()
    # save plot
    save_dir = os.path.dirname(logger.fig_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(logger.fig_path)


def train_qla_hyperparameters(args):
    """
    Faire un tableau avec toutes les differentes valeurs de alpha, beta, gamma
    :param args:
    :return:
    """
    alphas = [0.001, 0.01, 0.1, 0.2]  # (learning rate) small values -> luck parameter: we don't want to learn too fast.
    gammas = [0.1, 0.3, 0.5, 0.8]  # (discount) big values -> reward only at the end of the game
    epsilons = [0.5, 0.8, 0.9, 0.99]  # (exploration) big values -> many possibilities

    # Check whether gpu is available
    device = get_device()

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )
    best_score = (-np.inf, (None, None, None))
    for alpha in alphas:
        for epsilon in epsilons:
            for gamma in gammas:

                # Initialize the agent and use random agents as opponents
                agent = QLA_Poker(
                    epsilon=epsilon,
                    alpha=alpha,
                    gamma=gamma
                )
                # Start training
                if args.algorithm == "random":
                    test_agent = RandomAgent(num_actions=env.num_actions)
                elif args.algorithm == "V1":
                    test_agent = LeducHoldemRuleAgentV1()
                elif args.algorithm == "V2":
                    test_agent = LeducHoldemRuleAgentV2()
                else:
                    print("Unrecognize test agent")
                    return
                env.set_agents([agent, test_agent])
                for episode in range(args.num_episodes):
                    # Generate data from the environment
                    trajectories, payoffs = env.run(is_training=True)

                    # Reorganaize the data to be state, action, reward, next_state, done
                    trajectories = reorganize(trajectories, payoffs)

                    # Feed transitions into agent memory, and train the agent
                    # Here, we assume that DQN always plays the first position
                    # and the other players play randomly (if any)
                    for ts in trajectories[0]:
                        agent.feed(ts)
                score = tournament(env, args.num_games)[0]
                if best_score[0] < score:
                    best_score = (score, (alpha, gamma, epsilon))
    print("The best permutation was obtained by the parameters:\n"
          "alpha : " + str(best_score[1][0]) + \
          "\ngamma : " + str(best_score[1][1]) + \
          "\nepsilon : " + str(best_score[1][2]) + \
          "\nwith score of " + str(best_score[0]))

