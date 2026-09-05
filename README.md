# Deep Q-Network (DQN) — CartPole-v1

A PyTorch implementation of Deep Q-Network (DQN) reinforcement learning, trained to solve the classic `CartPole-v1` environment from Gymnasium.

## What This Does

The agent learns to balance a pole on a moving cart by choosing one of two actions at each step — push left or push right — based only on the current state (cart position, cart velocity, pole angle, pole angular velocity). It learns purely through trial and error, using rewards (+1 for every step the pole stays upright) as its only feedback signal.

## How It Works

This implementation includes the core components of DQN:

- **Q-Network** — a small feedforward neural network (2 hidden layers, 128 units each) that estimates the expected future reward (Q-value) for each possible action given the current state.
- **Experience Replay Buffer** — stores past transitions `(state, action, reward, next_state, done)` and samples random batches from it during training, which breaks harmful correlations between consecutive experiences and stabilizes learning.
- **Target Network** — a separate, slowly-updated copy of the Q-network used to compute stable training targets, preventing the network from chasing a constantly moving target.
- **Epsilon-Greedy Exploration** — the agent acts randomly with probability epsilon (which decays linearly over the first 400 episodes) and otherwise picks the action with the highest estimated Q-value, balancing exploration and exploitation.

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden layer size | 128 |
| Learning rate | 1e-3 |
| Discount factor (gamma) | 0.99 |
| Batch size | 32 |
| Replay buffer capacity | 100,000 |
| Target network update frequency | every 100 training steps |
| Epsilon decay | linear, 1.0 → 0.01 over 400 episodes |
| Target score (solved) | average reward ≥ 475 over 100 episodes |

## Requirements

```bash
pip install gymnasium torch matplotlib numpy
pip install "gymnasium[classic-control]"
```

## Running It

```bash
python python.py
```

The script will:
1. Train the DQN agent for up to 500 episodes (stopping early if it solves the environment)
2. Plot the reward per episode plus a moving average
3. Run 100 test episodes with a fully greedy (no exploration) policy and print mean/std/max/min scores

## Files

| File | Description |
|---|---|
| `python.py` | Full DQN implementation: network, replay buffer, agent, training loop, and evaluation |

## Notes

- Training uses GPU automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.
- The environment is considered "solved" when the average reward over the last 100 episodes reaches 475 or higher (out of a maximum possible 500 per episode).
