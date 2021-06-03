import numpy as np
import random
from collections import namedtuple, deque
from model_visual import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
#from prioritisedreplaybuffer import ReplayBuffer
from tools.replayBuffers import preBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.9            # discount factor
TAU = 0.5              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
B = 0.6                 # Importance Sampling factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on GPU" if torch.cuda.is_available() else "Running on CPU")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, input_channels, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.input_channels = input_channels
        self.action_size = action_size
        self.seed = seed
        self.frames = []
        random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(input_channels, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(input_channels, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR, momentum=0.95)
        
        # Replay memory
        self.memory = preBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def pre_process(self, image, reset=False):

        #print("Shape/Type", image.shape, type(image))
        image = image[:,40:,:,:]
        #print("Shape/Type", image.shape, type(image))    
        image = image.transpose((0, 3, 1, 2))

        if (len(self.frames) == 0) or reset:
            frames = [image]*4
        else:
            frames = frames[1:]
            frames.append(image)
        image = np.concatenate(frames, axis=1)
        #print("Shape/Type", image.shape, type(image))

        return image

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #print("check", self.t_step, len(self.memory))
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                indexes, TD_Error = self.learn(experiences, GAMMA)
                #print(TD_Error)
                self.memory.update(indexes, TD_Error)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(device) #.unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, indexes, weights = experiences
        #print("Learning from batch {}".format(states.shape))
        
        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"

        # Forward and backward passes
        Q_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_next * (1 - dones))
        
        Q_calculated = self.qnetwork_local(states).gather(1, actions)

        # Save TD Error for prioritise experience replay sampling
        TD_Error = Q_targets - Q_calculated

        # Calculate the loss function
        #loss = F.mse_loss(Q_calculated, Q_targets)
        #print(type(TD_Error), TD_Error.shape, type(weights), weights.shape)
        loss = (TD_Error * TD_Error * weights).mean() 

        #print("LOSS FUNCTION", loss, loss.shape)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)       

        return indexes, TD_Error              

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




