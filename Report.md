Unity ML-Agents
Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.
For game developers, these trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.
In this course, you will use Unity's rich environments to design, train, and evaluate your own deep reinforcement learning algorithms. You can read more about ML-Agents by perusing the GitHub repository.

The Environment
For this project, you will work with the Reacher environment.
 
Unity ML-Agents Reacher Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
Distributed Training
________________________________________
For this project, we will provide you with two separate versions of the Unity environment:
•	The first version contains a single agent.
•	The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like PPO, A3C, and D4PG that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
Solving the Environment
________________________________________
Note that your project submission need only solve one of the two versions of the environment.
Option 1: Solve the First Version
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.
Option 2: Solve the Second Version
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
•	After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
•	This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.
 
Plot of average scores (over all agents) with each episode.
The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.
The Environment
Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.
Step 1: Activate the Environment
________________________________________
If you haven't already, please follow the instructions in the DRLND GitHub repository to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.
(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.
Step 2: Download the Unity Environment
________________________________________
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:
Version 1: One (1) Agent
•	Linux: click here
•	Mac OSX: click here
•	Windows (32-bit): click here
•	Windows (64-bit): click here
Version 2: Twenty (20) Agents
•	Linux: click here
•	Mac OSX: click here
•	Windows (32-bit): click here
•	Windows (64-bit): click here
Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link (version 1) or this link (version 2) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)
Step 3: Explore the Environment
________________________________________
After you have followed the instructions above, open Continuous_Control.ipynb (located in the p2_continuous-control/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.
Watch the (silent) video below to see what kind of output to expect from the notebook (for version 2 of the environment), if everything is working properly! Version 1 will look very similar (where you'll see a single agent, instead of 20!).
Play Video
In the last code cell of the notebook, you'll learn how to design and observe an agent that always selects random actions at each timestep. Your goal in this project is to create an agent that performs much better!
Deep Deterministic Policy Gradient (DDPG)
This project implements an off-policy method called Deep Deterministic Policy Gradient and described in the paper Continuous control with deep reinforcement learning.
Below is taken from the article above.
Here we combine the actor-critic approach with insights from the recent success of Deep Q Network (DQN) (Mnih et al., 2013; 2015). Prior to DQN, it was generally believed that learning value functions using large, non-linear function approximators was difficult and unstable. DQN is able to learn value functions using such function approximators in a stable and robust way due to two innovations: 1. the network is trained off-policy with samples from a replay buffer to minimize correlations between samples; 2. the network is trained with a target Q network to give consistent targets during temporal difference backups. In this work we make use of the same ideas, along with batch normalization (Ioffe & Szegedy, 2015), a recent advance in deep learning.
Code implementation
The codes consist of 3 files:
•	model.py : the Actor and the Critic classes are deployed.
o	Actor - Build an actor (policy) network that maps states -> actions. 
o	Critic - Build a critic (value) network that maps (state, action) pairs -> Q-values.
•	ddpg_agent.py : Implement the DDPG agent, a Noise and a Replay Buffer class.
Hyperparameters:
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # Update the networks 10 times after every 20 timesteps
LEARN_NUMBER = 10       # Update the networks 10 times after every 20 timesteps
EPSILON = 1.0           # Noise factor
EPSILON_DECAY = 0.999999  # Noise factor decay

Below are the functions from Class agent:
Step:  Save experience in replay memory and use random sample from buffer to learn.
Act: Returns actions for given state as per current policy.
Learn: Update policy and value parameters using given batch of experience tuples.
Soft_updates:  Soft updates model parameters.

Below are functions from Class OUNoise (Ornstein-Uhlenbeck process):
Sample: Update internal state and return it as a noise sample.

Below are functions from Class ReplayBuffer:
Add: Add a new experience to memory
Sample:  Randomly sample a batch of  experiences from memory

Continuous_Control.ipynb : In this Jupyter Notebook file, we can train the agent. Below are the functions:
          Loads the environment:
           env =  UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')
send all actions to the environment
get next state (for each agent)
see if episode finished
Train 20 agents using DDPG
Plot the scores/rewards

For future work:
I like to explore A3C and A2C and compare the speed and accuracy of the three algorithms.


