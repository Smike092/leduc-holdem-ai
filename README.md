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
