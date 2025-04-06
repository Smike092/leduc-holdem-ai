import argparse
import os
from evaluate import evaluate_expectiminimax
from rl import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Project AI Leduc Holdem")
    parser.add_argument('--env', type=str, default='leduc-holdem')
    parser.add_argument('--algorithm', type=str, default='qla', choices=['cfr', 'qla', 'random', 'V1', 'V2',
                                                                         'expectiminimax'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/leduc_holdem_qla_result/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.algorithm == "qla":
        train_qla(args)
    elif args.algorithm == "cfr":
        train_cfr(args)
    elif args.algorithm in {'random', 'V1', 'V2'}:
        train_qla_hyperparameters(args)
    elif args.algorithm == "expectiminimax":
        evaluate_expectiminimax(args)
    else:
        print("Invalid Arguments")
