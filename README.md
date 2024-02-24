# rl-from-scratch

## Before getting into the fun stuff, heres a description about each file purpose
- `real-time-graph.py` is a demo for graphing a function in real time. This is useful for graphing live rewards (better visualization as opposed to reading numbers sequentially).
- `demo.py` lets you feel the simple mechanics of the RL environment we will use. It is a simple grid world where the agent can move up, down, left, right. The agent is rewarded for reaching the goal and penalized for hitting the wall. You will controller this with the arrow keys.
- `main.ipynb` is the main content in the video. Instead of writing about this here, I will explain it in the video and further in the README.md
- `main.py` is the same as main.ipynb but in a python file. This is useful for running the code in a terminal.

## Types of Machine Learning
- Supervised Learning: Given a dataset, the model learns to predict the output (ex. Next token prediction in GPTs). For example, given a dataset of images of cats and dogs, the model learns to predict if the image is a cat or a dog (ex. 0 if cat, 1 if dog).

- Unsupervised Learning: Given a dataset, the model learns to find patterns in the data. For example, given a dataset of images of cats and dogs, the model learns to cluster the images into two groups (cats and dogs). This is useful for finding patterns in data that is not explicitly labeled.

- Reinforcement Learning: Given an environment, the model learns to take actions to maximize the reward. For example, given a game, the model learns to take actions to maximize the score. This is useful for learning how to play games, control robots, and more. Unlike supervised learning, the model does not have access to the correct output. Instead, the model learns from trial and error. Therefore, this type of learning is not data driven, but rather experience driven.

## Reinforcement Learning
Pros:
- Doesn't require a massive (or small) dataset
- Can teach humans how to play games (AlphaGo: https://deepmind.com/research/case-studies/alphago-the-story-so-far)
- Optimize complex systems (ex. traffic lights, supply chain, etc.)

Cons:
- Can be slow to train (batch processing sometimes isn't possible)
- Can be hard to tune hyperparameters (use automated hyperparameter tuning)
- Can be hard to debug (use visualization tools)
- If the action space is large, it can be hard to explore all possible actions (use exploration strategies). For example, in Minecraft, the number of actions to pick from is huge at each frame. Say the agent has to break a block of wood as its first reward. The agent has to correctly select the optimal action (left mouse button) hundreds of times in a row to break the wood.

## Our Environment

We will use a simple grid world environment. The agent can move up, down, left, right. The agent is rewarded for reaching the goal and penalized for hitting the wall. The environment is a 5x5 grid. The agent (agent and player can be used interchangeably) and goal are initialized at random positions in the grid. The agent is penalized each time step (each time it makes a decision) rewarded for hitting the goal. I include a slightly more complex reward system for its decisions. Its rewarded as a function of delta distance from the goal. The closer the agent moves to the goal, the higher the reward. The further the agent moves from the goal, the lower the reward. This is a simple environment to understand the mechanics of RL. We will use this environment to understand the mechanics of RL and build a simple Q-learning agent.

## Q?

Deep Q-Learning VS Q-Learning:
- Q-Learning: Uses a table to store the Q-values. This is useful for small action spaces. The Q-value is the expected reward for taking an action in a state. 
- Deep Q-Learning: Uses a neural network to store the Q-values. This is useful for large action spaces. The Q-value is the expected reward for taking an action in a state.
- Both of the above use the Bellman equation to update the Q-values. The Bellman equation is a recursive equation that updates the Q-values based on the reward and the next state. The Q-value is updated to be the reward plus the discounted maximum Q-value of the next state. The discount factor is used to weigh the future rewards. This is useful for long term planning. The discount factor is usually set to 0.9 or 0.99. The discount factor is used to weigh the future rewards. This is useful for long term planning. The discount factor is usually set to 0.9 or 0.99.


