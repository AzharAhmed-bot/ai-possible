# Lab 3 — Q-Learning (Reinforcement Learning)

A tiny reinforcement-learning agent that learns to walk across a 2×3 grid to
reach a goal while avoiding a danger cell — using the **Q-learning** algorithm.

## What it does

- Sets up a 2×3 grid world. The agent starts top-left and must reach the
  bottom-right **goal** (+10 reward) while avoiding a **danger** cell (−10).
- Learns a **Q-table** (6 states × 4 actions) that scores how good each move is
  from each square.
- Trains for **20,000 episodes** using the Bellman update with an
  **epsilon-greedy** strategy (mostly picks the best-known move, occasionally
  explores a random one).
- Prints the final Q-table and the best action to take from every square.

## Key ideas (in plain terms)

- **State** = which square the agent is on. **Action** = up / down / left / right.
- **Reward** = the number in the square it lands on.
- **Q-value** = "how much total reward can I expect if I take this action here?"
- After enough practice the agent's Q-table points the way to the goal.

## Result

Average reward over the last 1,000 episodes: **≈ 1.66**. The learned policy
correctly steers the agent toward the goal and away from the danger cell.

## Plots (in [`plots/`](plots/))

| File            | What it shows                                                           |
|-----------------|-------------------------------------------------------------------------|
| `figure_01.png` | **Convergence curve** — rolling-average reward rising as the agent learns. |
| `figure_02.png` | **Q-table heatmap** — the learned value of every action in every state. |

## Run it

```bash
python3 Lab3_Q_Learning.py
```

Plots are saved to `plots/`.
