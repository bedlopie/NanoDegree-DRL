[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### File Structure

File structure

```bash
- tools
|- binaryTreeSearch.py              # Binary Tree Search Support
|- hyperParameters.py               # HyperParameters Support
|- replayBuffers.py                 # All replay buffers Support

- README.md                         # Current file (Markdown)
- model.py                          # Deep Neural network definition
- dqn_agent.py                      # Agent code, act and learn
- Navigation.ipynb                  # Main UI interface for training and using
- checkpoint_vector_max.pth         # Saved model with Max score
```

### Instructions

Follow the instructions in `Navigation.ipynb` to get started.

This implementation is allowing Deep Reinforcement learning with 3 optional features
    - Double DQN
    - Prioritised Experience Replay
    - Dueling

You can manage the learning parameters of the Agent in the cell 7 of the `Navigation.ipynb` file.

    ## hyperparameters = 
    # Those are the general learning parameters
    'dqn' : 
        'n_episodes':          1300,            # Number of episode of self training
        'max_t':               1000,            # Max value of steps per episode (should be more than 200)
        'epsilon':                              # Epsilon in Epsilon Greedy
                               { 'name': 'ε', 'profil': 'geometric', 'init': 1., 'bound': 0.01, 'steps': 300 }
  
    # Those are the Agent parameters
    'agent' :
        'GPU':                 True,            # Use GPU (Bool)
        'state_size':          37,              # State Size, here 37
        'action_size':         4,               # Action space, here 4
        'seed':                0,               # Random generator seed
        'DDQN':                True,            # Use Double DQN (Bool)
        'Prioritised_replay':  False,           # Use Prioritised Experience Replay (Bool)
        'Dueling':             True,            # Use Dueling model (Bool)
        'BUFFER_SIZE':         int(1e5),        # replay buffer size
        'BATCH_SIZE':          64,              # minibatch size
        'GAMMA':               0.99,            # discount factor
        'LR':                  5e-2,            # learning rate 
        'LEARN_EVERY':         4,               # how often to update the network
        'UPDATE_EVERY':        20,              # how often to update the network
        'alpha' :                               # Alpha parameter in PER (Schocastic vs TD error importance selection)
                               { 'name': 'α', 'profil': 'constant', 'init': 0.7 },
        'TD_Error_clip':       [0., 1.],        # Cliping of TD Error range in PER selection
        'beta' :                                # Beta parameter in PER (integration of Wi parameter)
                               { 'name': 'β', 'profil': 'linear', 'init': 0.6, 'bound': 1., 'steps': 200*75 },
        'tau':                                  # Parameter of learning from fixed policy to 
                               { 'name': 'τ', 'profil': 'constant', 'init': 5e-2 },

    ## Parameters like Epsilon Alpha Beta Tau can me fixed or can follow a parametric value.
    
    # Use a dictionary to pass parameters : Example here with Epsilon
        'name': 'ε',                # Display name in logs
        'profil': 'geometric',      # profil of evolution : Geometric (multiplication is q), linear (application of q), constant (no evolution)
        'init': 1.,                 # Starting value
        'bound': 0.01,              # End value
        'steps': 300                # Number of steps to move from Start to End Value.

    ## Model
    The Deep Neural Network can be modified in model.py

Checkpoint have been save for the current model structure. Naming convention is checkpoint_vector_##.pth (## average score obtained)
checkpoint_vector_max.pth being the checkpoint saved at the end of 1300 episodes.

### Improvement

    On the vector problem:
        - find a better quicker NN structure that converge and learn faster.
        - find better alpha and beta parameter that make PER more beneficial.

    On the vision problem:
        - chnage the input processing of the input image. Stop using RGB and create 1 frame for Yellow, one for Blue and one for background, on 4 frames, so 12 input channel to the convolutional network.
        - understand how long a network need to converge. (not like deepmind brute force)
        - better understand the link between structure and convergence.
        - better understand impact/benfits of more depth or more convolutions.
        - better understand where to put residual layers.

    I have spent some time on the code to make it nice, it is more now the fine tunning of parameters the next and perpetual challenge.

### Challenge: Learning from Pixels

I have tried to solve the vision agent robots, but I ws never able to find a set of paramerters or deep neural network that can converge and learn.
I guess that the size of the input (in pixel) make the task difficult, and preprocessing of the image is key to try to simplify the task of the CNN

### Instruction for Learning from Pixels

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
