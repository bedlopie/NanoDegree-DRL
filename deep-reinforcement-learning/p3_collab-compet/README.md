[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[final]: https://github.com/bedlopie/NanoDegree-DRL/blob/58979e76eadc766092407e686138289e42179741/deep-reinforcement-learning/p3_collab-compet/Tennis.gif "Trained"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][final]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Installation

This repository is quite sensible to package version.
* You need Python 3.8.x (not that I couldn't make it waork on Python 3.9 at the date of creation of this file)
* Then you need to clone this repository
```bash
git clone https://github.com/bedlopie/NanoDegree-DRL.git
```
* I would advise you to use a virtual environement to make sure all is well installed and isolated from the rest of your machine, don't forget to upgrade pip
```bash
python -m venv drlnd
python.exe -m pip install --upgrade pip
```
* Then pip install the requirement.txt file in this directory
```bash
pip install -r requirements.txt
```
* Then, you need to install another set of package
```bash
pip install -r ./deep-reinforcement-learning/python/requirements.txt
```
* And last create the Kernel to run all of this, you need to set jupyter to use this kernel instead of the default one
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

Then you can go in project 1 - 2 - 3 to run the projects

### How to use the project

**The projects are all built the same way.**
You have 3 files
1. *.ipynb           / file to run project (ie. Tennis.ipynb)
2. model.py          / file containing the Deep Neural Network model 
3. maddpg_agent.py   / file containing the agent in charge of acting on the environement
4. tools.py          / file containing Replay buffer and Reward buffer

model and agent file have been setup and fine tune.
you need to go in the ipynb file to run the project.

#### Demo vs Training mode

in cell 2, you have a "demo" variable. Set it to True to view save model and to False to train a model

#### Training

set up hyperparameters of agent and model in cell number 8, by modifying the dictonary called hyperparameters
Checkpoints are created along the way. One for critic network and one for actor
* checkpoint_agent#_critic_episode##.pth         ## being the average score value obtained in training or the score obtained
* checkpoint_agent#_critic_solved.pth            checkpoint created when problem is considered solved
* checkpoint_agent#_critic_max.pth               checkpoint created when training is finished (might not be the best)



### Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  
