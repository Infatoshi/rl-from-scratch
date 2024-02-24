# rl-from-scratch


## Introduction (1 min)
Hey everyone, welcome back. My name is Elliot and in this video, we will be learning about reinforcement learning from scratch. I assume you have heard of the term RL and have a general understanding enough to get you through this video. 
My logic for creating this is the limited content on the internet that explains RL from scratch in a shorter time frame. I will be uploading another video soon where we comprehensively build up the codebase from nothing. It may already be up by the time you are watching this.
This video is particularly designed for intermediate and beginner level programmers. If you are an expert, you might find this video a bit slow. If you are a beginner, you might find this video a bit fast. I will try to balance the two. We will we using python. Lets get started.

## Before getting into the fun stuff, heres a description about each file purpose (0.5 mins)
- `real-time-graph.py` is a demo for graphing a function in real time. This is useful for graphing live rewards (better visualization as opposed to reading numbers sequentially).
- `demo.py` lets you feel the simple mechanics of the RL environment we will use. It is a simple grid world where the agent can move up, down, left, right. The agent is rewarded for reaching the goal and penalized for hitting the wall. You will controller this with the arrow keys.
- `main.ipynb` is the main content in the video. Instead of writing about this here, I will explain it in the video and further in the README.md
- `main.py` is the same as main.ipynb but in a python file. This is useful for running the code in a terminal.

## Types of Machine Learning (tell viewers to refer to the README.md for more details, I assume they already know this to my level of explanation)
- Supervised Learning: Given a dataset, the model learns to predict the output (ex. Next token prediction in GPTs). For example, given a dataset of images of cats and dogs, the model learns to predict if the image is a cat or a dog (ex. 0 if cat, 1 if dog).

- Unsupervised Learning: Given a dataset, the model learns to find patterns in the data. For example, given a dataset of images of cats and dogs, the model learns to cluster the images into two groups (cats and dogs). This is useful for finding patterns in data that is not explicitly labeled.

- Reinforcement Learning: Given an environment, the model learns to take actions to maximize the reward. For example, given a game, the model learns to take actions to maximize the score. This is useful for learning how to play games, control robots, and more. Unlike supervised learning, the model does not have access to the correct output. Instead, the model learns from trial and error. Therefore, this type of learning is not data driven, but rather experience driven.

## Reinforcement Learning (1 min)
Pros:
- Doesn't require a massive (or small) dataset
- Can teach humans how to play games (AlphaGo: https://deepmind.com/research/case-studies/alphago-the-story-so-far)
- Optimize complex systems (ex. traffic lights, supply chain, etc.)

Cons:
- Can be slow to train (batch processing sometimes isn't possible)
- Can be hard to tune hyperparameters (use automated hyperparameter tuning)
- Can be hard to debug (use visualization tools)
- If the action space is large, it can be hard to explore all possible actions (use exploration strategies). For example, in Minecraft, the number of actions to pick from is huge at each frame. Say the agent has to break a block of wood as its first reward. The agent has to correctly select the optimal action (left mouse button) hundreds of times in a row to break the wood.

## Our Environment (1 min)

We will use a simple grid world environment. The agent can move up, down, left, right. The agent is rewarded for reaching the goal and penalized for hitting the wall. The environment is a 5x5 grid. The agent (agent and player can be used interchangeably) and goal are initialized at random positions in the grid. The agent is penalized each time step (each time it makes a decision) rewarded for hitting the goal. I include a slightly more complex reward system for its decisions. Its rewarded as a function of delta distance from the goal. The closer the agent moves to the goal, the higher the reward. The further the agent moves from the goal, the lower the reward. This is a simple environment to understand the mechanics of RL. We will use this environment to understand the mechanics of RL and build a simple Q-learning agent.

## Q? (3 mins)

### Deep Q-Learning VS Q-Learning:
- *Q-Learning:* Uses a table to store the Q-values. This is useful for small action spaces. The Q-value is the expected reward for taking an action in a state. 
- *Deep Q-Learning:* Uses a neural network to store the Q-values. This is useful for large action spaces. The Q-value is the expected reward for taking an action in a state.
- Both of the above use the Bellman equation to update the Q-values. The Bellman equation is a recursive equation that updates the Q-values based on the reward and the next state. The Q-value is updated to be the reward plus the discounted maximum Q-value of the next state. The discount factor is used to weigh the future rewards. This is useful for long term planning. The discount factor is usually set to 0.9 or 0.99. The discount factor is used to weigh the future rewards. This is useful for long term planning.

## General Structure of this example (2 mins)
1. We have an environment with some rules, rewards (as defined above) and an action space (action space is a fancy way of saying the possible actions the agent can take).
2. We have an agent that interacts with the environment. The agent takes actions in the environment and receives rewards. The agent uses the rewards to learn the best actions to take in the environment.
3. We have a neural network that we optimize to minimize the difference between the predicted Q-values and the actual Q-values. The Q-values are the expected rewards for taking an action in a state. The neural network takes the state as input and outputs the Q-values for each action. I'm not going to go into the details of the neural network in this video since its time consuming and not the focus of this video. If this strikes your interest, I have a video on neural networks just for you: https://www.youtube.com/watch?v=Gk_5I6YG_Jw&t=
4. To fancy it up a bit, I implemented a feature called epsilon decay. 
Put simply, epsilon decay is a way to balance exploration and exploitation. Exploration is the process of trying new things. Exploitation is the process of using what you already know to get rewards. Epsilon decay is a way to balance these two processes. At the start, the agent is more likely to explore. As the agent learns, the agent is more likely to exploit. This is useful for learning the environment and then using what you know to get rewards. Once the agent has learned the environment, the agent is more likely to exploit. This is useful for getting the most rewards.


## Conclusion (1 mins)
For those wondering, yes, I do 1-on-1 tutoring and consulting. If you're interested, feel free to shoot me a DM on Fiverr, Twitter, LinkedIn, or Discord. All links will be in the video description and bottom of the README.md in the repo.

If you found this video/repo helpful, you should definitely subscribe. It not only supports me, but also tells the Youtube algorithm that this content is helpful and therefore recommends it to more people. If you want to see more content like this, please let me know. I'm always looking for new ideas. Thanks for watching and I'll see you in the next video.

## Links
*Fiverr* - https://www.fiverr.com/elliotarledge

*Twitter* - https://twitter.com/elliotarledge

*LinkedIn* - https://www.linkedin.com/in/elliot-arledge-a392b7243/

*Discord* - https://discord.gg/K7Jd4rFNcW

*Youtube* - https://www.youtube.com/channel/UCjlt_l6MIdxi4KoxuMjhYxg