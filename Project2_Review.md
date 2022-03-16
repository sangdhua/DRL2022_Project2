Meets Specifications
Dear Udacian, the project is very well implemented and meets the specifications! Congratulations on implementing Deep Deterministic Policy Gradients algorithm and successfully training the agent. The agent solves the environment and the performance is pretty good. It achieves an average score of +30.1 from 22-122 episodes only. Impressive work! The actor and critic networks seem to be very efficient with two hidden layers each. Good work using relu activation in the hidden layers of the actor and critic networks. Good work using batch normalization also. A very good submission overall.

As next step, please go through the resource: human-like robot hand trained to manipulate physical objects with unprecedented dexterity.

All the best for the next project! :smile:

Training Code
The repository includes functional, well-documented, and organized code for training the agent.

The repository contains Continuous_Control.ipynb notebook, code files, readme, and project report. The code is functional.
Great job solving the environment using Deep Deterministic Policy Gradients.

The code is written in PyTorch and Python 3.

The code is written in pytorch and python3.

A good read: PyTorch vs TensorFlow — spotting the difference
The submission includes the saved model weights of the successful agent.

Thanks for including the saved model weights of the successful agent. The weights are saved as following files:

actor_checkpoint.pth
critic_checkpoint.pth
README
The GitHub submission includes a README.md file in the root of the repository.

Submission includes README file for the project.

The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved).

Great work providing the details of the project environment in the README file:

The Introduction section describes the state space, action space, and the reward function.
The Distributed Training section explains about the different versions of the environments.
The Solving the Environment section explains about when the environments are considered solved.
The README has instructions for installing dependencies or downloading needed files.

Proper instructions have been specified in the The Environment section of the README to install the dependencies and download the necessary files.

The README describes how to run the code in the repository, to train the agent. For additional resources on creating READMEs or using Markdown, see here and here.

README file contains the instructions to train the agent using Continuous_Control.ipynb file.

Report
The submission includes a file in the root of the GitHub repository (one of Report.md, Report.ipynb, or Report.pdf) that provides a description of the implementation.

Report has been included in the root of the github repository.

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

The report is rather informative providing an insight on every aspect of the project which includes implementation, model architectures, hyperparameters, rewards, future works.

Good choice to implement the DDPG algorithm.
It is found to work very well with continuous action space.
Good implementation of the Actor and Critic networks.
Good work keeping two hidden layers in the actor and critic networks, respectively.
Good work using batch normalization for improved performance.
Good decision choosing to use the relu activation in both the networks.
Good decision to use replay buffer to store and recall experience tuples.
Good job using the target networks for Actor and Critic, as suggested in the original paper.
Good choice to use tau to update the target network.
A plot of rewards per episode is included to illustrate that either:

[version 1] the agent receives an average reward (over 100 episodes) of at least +30, or
[version 2] the agent is able to receive an average reward (over 100 episodes, and over all 20 agents) of at least +30.
The submission reports the number of episodes needed to solve the environment.

Awesome
The agent seems to perform very well!
The agent is able to achieve an average score of +30.1 over last 100 episodes in from 22-122 episodes!
The submission discusses the rewards plot obtained clearly.

Screenshot from 2022-01-10 15.02.00.png

The submission has concrete future ideas for improving the agent's performance.

Great job providing the ideas to experiment more in future with the project by trying:

Advantage Actor Critic algorithm
Asynchronous Advantage Actor Critic algorithm
Suggestions
You should try implementing Prioritized Experience Replay also. It helps to improve the performance and significantly reduces the training time. This should also help to stabilize the performance to some extent. A fast implementation of Prioritized Experience Replay is possible using a special data structure called Sum Tree. I found a good implementation here.

Below is a comparison of DDPG with random sampling vs DDPG with PER for the Reacher environment. It's quite evident how episode variation decreased and performance improved.

PER vs Random sampling.png

Also, I request you to check the following posts to get familiar with more reinforcement learning algorithms.

Asynchronous Actor-Critic Agents (A3C)
Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO)
Here is an implementation of PPO on tennis environment. The training was slow but the final average score achieved was almost 1.25 (with some fluctuation). You should surely try PPO in future.
