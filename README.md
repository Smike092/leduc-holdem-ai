# 🤖 Leduc Hold'em AI Agents

Implementation of intelligent agents (Q-Learning, CFR, Expectiminimax) to learn and play **Leduc Hold’em Poker** using reinforcement learning and game theory. Built on top of the [RLCard](https://github.com/datamllab/rlcard) framework.

## 🎯 Project Goals

- Explore simplified poker strategy with reinforcement learning.
- Compare different AI approaches: Q-Learning, Counterfactual Regret Minimization (CFR), and Expectiminimax.
- Evaluate each agent's performance against distinct poker profiles: Random, Aggressive, and Cautious players.

## 🧠 Implemented Agents

| Agent              | Type                       | Description                                                                 |
|--------------------|----------------------------|-----------------------------------------------------------------------------|
| **QLA_Poker**      | Reinforcement Learning     | Learns Q-values of state-action pairs using temporal difference updates.   |
| **CFR Agent**      | Self-play Algorithm        | Minimizes regret to converge towards a Nash equilibrium strategy.          |
| **Expectiminimax** | Tree Search with Heuristics| Uses search with chance nodes and evaluation heuristics.                   |

## 🃏 Game: Leduc Hold'em

- 2-player simplified poker
- 6 cards total: J, Q, K (♠️ & ♥️)
- 1 private card each + 1 public card
- 2 betting rounds with limited raise counts

## 🛠️ Setup Instructions

1. **Clone the repo**:
   git clone https://github.com/your-username/leduc-holdem-ai-agents.git

   cd leduc-holdem-ai-agents
   
2. **Install RLCard**:
    
    pip install rlcard

3. **Replace RLCard game engine** (only for Expectiminimax):
    
    cp game_expectimax.py YOUR_ENV_PATH/rlcard/games/leducholdem/game.py

## 🚀 Running the Agents

### Train Q-Learning agent:
python main.py --algorithm qla --num_episodes 100000 --evaluate_every 5000

### Train CFR agent:
python main.py --algorithm cfr --num_episodes 100000 --evaluate_every 5000

### Evaluate Expectiminimax:
python main.py --algorithm expectiminimax --num_games 10000

### HyperParameter Search:
python main.py --algorithm V1 --num_episodes 20000


📈 Results

Agents were tested against 3 profiles:

    Random: Chooses actions uniformly

    Aggressive: Prefers raising

    Cautious: Only plays top-ranked hands

Performance was measured as average number of big blinds won.

    📊 Example result plot is saved in experiments/expectiminimax.png
    
