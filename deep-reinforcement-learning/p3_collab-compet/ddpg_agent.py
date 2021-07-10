import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor_flex, Critic_flex
from tools import OUNoise, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, GPU, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY, clip_gradient):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        def torch_device(GPU_flag):
            if GPU_flag == True:
                if torch.cuda.is_available():
                    print("Running on GPU")
                    return torch.device("cuda:0")

            print("Running on CPU")
            return torch.device("cpu")

        self.state_size = state_size
        self.action_size = action_size
        self.id = id
        self.seed = random_seed
        random.seed(random_seed)
        self.device = torch_device(GPU)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_flex(state_size, action_size, random_seed, hidden_layer=Actor_network).to(self.device)
        self.actor_target = Actor_flex(state_size, action_size, random_seed, hidden_layer=Actor_network).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_flex(state_size, action_size, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_target = Critic_flex(state_size,action_size, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, random_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.LEARN_EVERY = LEARN_EVERY
        self.LEARN_REPEAT = LEARN_REPEAT
        self.BUFFER_SIZE = BUFFER_SIZE                  # replay buffer size
        self.BATCH_SIZE = BATCH_SIZE                    # minibatch size
        self.GAMMA = GAMMA                              # discount factor
        self.TAU = TAU                                  # for soft update of target parameters
        self.LR_ACTOR = LR_ACTOR                        # learning rate of the actor
        self.LR_CRITIC = LR_CRITIC                      # learning rate of the critic
        self.WEIGHT_DECAY = WEIGHT_DECAY                # L2 weight decay
        self.clip_gradient = clip_gradient
        
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = self.t_step + 1
        if self.t_step % self.LEARN_EVERY == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.LEARN_REPEAT):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(np.array(action), -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (self.clip_gradient):
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save(self, id, value):
        torch.save(self.actor_local.state_dict(), 'checkpoint_agent{}_actor_episode{}.pth'.format(id, value))
        torch.save(self.critic_local.state_dict(), 'checkpoint_agent{}_critic_episode{}.pth'.format(id, value))

    def load(self, id, value):
        self.actor_local.load_state_dict(torch.load('checkpoint_agent{}_actor_episode{}.pth'.format(id, value)))
        self.critic_local.load_state_dict(torch.load('checkpoint_agent{}_critic_episode{}.pth'.format(id, value)))    