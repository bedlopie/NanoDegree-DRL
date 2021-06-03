import numpy as np
import random
from collections import namedtuple, deque, OrderedDict
from model import QNetwork
from tools.replayBuffers import preBuffer, replayBuffer
from tools.hyperParameters import geometricParameter, linearParameter, Hyperparameters

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, DDQN, Prioritised_replay, Dueling, state_size, action_size, seed, BUFFER_SIZE, BATCH_SIZE, GAMMA, tau, LEARN_EVERY, UPDATE_EVERY, LR, GPU, alpha, TD_Error_clip, beta):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        def torch_device(GPU_flag):
            if GPU_flag == True:
                if torch.cuda.is_available():
                    print("Running on GPU")
                    return torch.device("cuda:0")

            print("Running on CPU")
            return torch.device("cpu")

        self.DDQN = DDQN
        self.Prioritised_replay = Prioritised_replay
        self.Dueling = Dueling
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.tau = Hyperparameters(**tau)
        self.UPDATE_EVERY = UPDATE_EVERY
        self.LEARN_EVERY = LEARN_EVERY
        self.LR = LR
        self._device = torch_device(GPU)
        random.seed(self.seed)

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed, self.Dueling).to(self._device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed, self.Dueling).to(self._device)
        self.optimizer = optim.SGD(self.qnetwork_local.parameters(), lr=self.LR, momentum=0.01, nesterov=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[800, 1500, 3000, 4500], gamma=0.1)

        # Replay memory
        if Prioritised_replay:
            self.memory = preBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self._device, alpha, TD_Error_clip, beta, self.seed)
        else:
            self.memory = replayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, self._device, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # loss buffer when training
        self.losses = []
    


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        #print(len(self.memory), BATCH_SIZE)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = self.t_step + 1
        if self.t_step % self.LEARN_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                #print(experiences)
                indexes, TD_Error = self.learn(experiences, self.GAMMA)
                if indexes != []:
                    self.memory.update(indexes, TD_Error)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
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
        if self.Prioritised_replay:
            states, actions, rewards, next_states, dones, indexes, weights  = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            indexes = []
            weights = []

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
 
        if self.DDQN:

            # Get value from local network at next states, to get max action
            self.qnetwork_local.eval()
            with torch.no_grad():
                local_nextstates_actions = self.qnetwork_local(next_states).detach()
            self.qnetwork_local.train()
            #print("Size", local_nextstates_actions.size())
            #print("Example", local_nextstates_actions)
            #print("ArgMax", torch.argmax(local_nextstates_actions, axis=1, keepdim=True))
            # Find Max action to be done at next states
            maxaction_local_nextstates = torch.argmax(local_nextstates_actions, axis=1, keepdim=True)
            # Calculate Q value from target network using max action from local network
            Q_next = torch.gather(self.qnetwork_target(next_states).detach(), 1, maxaction_local_nextstates)
            #print("Q value", Q_next)
            #print(Q_next)
            # Calculate from reward and Q value at nest States the current state value 
            Q_targets = rewards + (gamma * Q_next * (1 - dones))

        else:
            # Forward and backward passes
            Q_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            #print(Q_next)
            Q_targets = rewards + (gamma * Q_next * (1 - dones))


        # Calculate the local network value for the states
        Q_calculated = self.qnetwork_local(states).gather(1, actions)
        
        # Save TD Error for prioritise experience replay sampling
        TD_Error = Q_targets - Q_calculated


        # Calculate the loss function
        if self.Prioritised_replay:
            #print(type(TD_Error), TD_Error.shape, type(weights), weights.shape)
            if TD_Error.size() != weights.size():
                print("ERROR", TD_Error.size(), weights.size())
                print(TD_Error)
                print(weights)
            loss = F.mse_loss(Q_calculated*(weights**0.5), Q_targets*(weights**0.5))    
            #loss = (TD_Error * TD_Error * weights).mean()
        else:
            loss = F.mse_loss(Q_calculated, Q_targets)
        #print("LOSS FUNCTION", loss, loss.shape)
        #print(loss)
        self.losses.append(loss.detach().cpu().tolist())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        if self.t_step % self.UPDATE_EVERY == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau.value)        

        # update agent parameters
        # if self.Prioritised_replay:
        #     self.memory.alpha.next()
        #     self.memory.beta.next()
        # self.tau.next()

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

    def display_parameters(self):
        if self.Prioritised_replay:
            return "{} {}".format(self.memory.alpha.display(), self.memory.beta.display())
        return ""