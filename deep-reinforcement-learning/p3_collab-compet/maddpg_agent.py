import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor_flex, Critic_flex_maddpg

import torch
import torch.nn.functional as F
import torch.optim as optim

class MADDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, nb_agent, id, random_seed, device, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.id = id
        self.action_size_maddpg = action_size * nb_agent
        self.seed = random_seed
        random.seed(random_seed)
        self.device = device

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_flex(state_size, action_size, random_seed, hidden_layer=Actor_network).to(self.device)
        self.actor_target = Actor_flex(state_size, action_size, random_seed, hidden_layer=Actor_network).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_flex_maddpg(state_size, self.action_size_maddpg, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_target = Critic_flex_maddpg(state_size, self.action_size_maddpg, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

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
    

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for state in states:
            state = torch.from_numpy(state).float().to(self.device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            if add_noise:
                action += self.noise.sample()
            actions.append(action)    
        return np.clip(np.array(actions), -1, 1)

    def reset(self):
        self.noise.reset()

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


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(loc=0, scale=1, size=len(x))
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
        self.seed = seed
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class MADDPG_Agency():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, nb_agent, random_seed, GPU, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY):
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
        self.nb_agent = nb_agent
        self.seed = random_seed
        random.seed(random_seed)
        self.device = torch_device(GPU)

        # Setup of Agents
        self.agents = [MADDPG_Agent(state_size, action_size, nb_agent, id, random_seed, self.device, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY) for id in range(nb_agent)]

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

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        for i, state in enumerate(states):
            state = torch.from_numpy(state).float().to(self.device)
            self.agents[i].actor_local.eval()
            with torch.no_grad():
                action = self.agents[i].actor_local(state).cpu().data.numpy()
            self.agents[i].actor_local.train()
            if add_noise:
                action += self.agents[i].noise.sample()
            actions.append(action)    
        return np.clip(np.array(actions), -1, 1)

    def reset(self):
        for i in range(self.nb_agent):
            self.agents[i].noise.reset()

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
        
        actions_next = None
        for i in range(self.nb_agent):
            action_next = self.agents[i].actor_target(next_states[i])
            if actions_next == None:
                actions_next = action_next
            else:
                actions_next = torch.cat((actions_next, action_next), 1)    

        for i in range(self.nb_agent):
            Q_targets_next = self.agents[i].critic_target(next_states[i], actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.agents[i].critic_local(states[i], actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.agents[i].critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].critic_local.parameters(), 1)
            self.agents[i].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = None
        for i in range(self.nb_agent):
            action_next = self.agents[i].actor_local(states[i])
            if actions_pred == None:
                actions_pred = action_next
            else:
                actions_pred = torch.cat((actions_pred, action_next), 1)    

        for i in range(self.nb_agent):
            actor_loss = -self.agents[i].critic_local(states[i], actions_pred).mean()
            # Minimize the loss
            self.agents[i].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.agents[i].actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.agents[i].critic_local, self.agents[i].critic_target, self.TAU)
            self.soft_update(self.agents[i].actor_local, self.agents[i].actor_target, self.TAU)                     

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
