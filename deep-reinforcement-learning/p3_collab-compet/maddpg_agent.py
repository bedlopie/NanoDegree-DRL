import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor_flex, Critic_flex
from tools import OUNoise, ReplayBuffer, RewardBuffer
from tools import _debug, _assert, _info, _whatis

import torch
import torch.nn.functional as F
import torch.optim as optim


def extract_agent_data(data, index):
    '''
    Imply that the torch tensor shape of input data is (N, P, *).
    N being the number of examples (batchsize)
    P the player dimension, (here we want the "index" value of that dimension)
    output is shape (N, *)
    '''
    return data.transpose(0, 1)[index]


class MADDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, nb_agent, id, random_seed, device, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY, clip_gradient):
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
        self.actor_scheduler = optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=[750, 1200, 2000, 3000], gamma=0.1)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_flex(state_size * nb_agent, self.action_size * nb_agent, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_target = Critic_flex(state_size * nb_agent, self.action_size * nb_agent, random_seed, hidden_layer=Critic_network).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_scheduler = optim.lr_scheduler.MultiStepLR(self.critic_optimizer, milestones=[750, 1200, 2000, 3000], gamma=0.1)

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
        self.clip_gradient = clip_gradient
    

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
        return np.clip(np.array(actions), -1., 1.)

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

    def learn(self, agent_id, experiences, gamma, actions_next, actions_pred):

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        #_whatis(next_states, "Next States all agents")
        #_whatis(actions_next, texte="Next actions", block=True)
        torch.autograd.set_detect_anomaly(True)
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states.flatten(start_dim=1), actions_next)
        #_whatis(Q_targets_next, texte="Q target NEXT shpae should be [512]")
        # Compute Q targets for current states (y_i)
        reward = torch.unsqueeze(extract_agent_data(rewards, agent_id) ,dim=1)
        #_whatis(reward, texte="reward shape should be [512]")
        done = torch.unsqueeze(extract_agent_data(dones, agent_id) ,dim=1)
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        #_whatis(Q_targets, texte="Q target shape should be [512]")
        # Compute critic loss
        Q_expected = self.critic_local(states.flatten(start_dim=1), actions.flatten(start_dim=1))
        #_whatis(Q_expected, texte="Q expected shpae should be [512]", block=True)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        #print("Going Backward, agent ", self.id)
        critic_loss.backward()
        if (self.clip_gradient):
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        
        actor_loss = -self.critic_local(states.flatten(start_dim=1), actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)

    def save(self, value):
        torch.save(self.actor_local.state_dict(), 'checkpoint_agent{}_actor_episode{}.pth'.format(self.id, value))
        torch.save(self.critic_local.state_dict(), 'checkpoint_agent{}_critic_episode{}.pth'.format(self.id, value))

    def load(self, value):
        self.actor_local.load_state_dict(torch.load('checkpoint_agent{}_actor_episode{}.pth'.format(self.id, value)))
        self.critic_local.load_state_dict(torch.load('checkpoint_agent{}_critic_episode{}.pth'.format(self.id, value)))      


class MADDPG_Agency():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, nb_agent, random_seed, GPU, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY, clip_gradient):
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
        self.agents = [MADDPG_Agent(state_size, action_size, nb_agent, id, random_seed, self.device, LEARN_EVERY, LEARN_REPEAT, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, Actor_network, LR_ACTOR, Critic_network, LR_CRITIC, WEIGHT_DECAY, clip_gradient) for id in range(nb_agent)]
        self.optimized  = False

        # Replay memory
        self.buffer = RewardBuffer(10, GAMMA, self.device, random_seed)
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
        self.memory.add_experience(self.buffer.add(states, actions, rewards, next_states, dones))

        #_info(next_states, "Next_state for one step (add to memory)")

        # Learn every UPDATE_EVERY time steps.
        self.t_step = self.t_step + 1
        if self.t_step % self.LEARN_EVERY == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.BATCH_SIZE:
                for _ in range(self.LEARN_REPEAT):
                    for agent_id in range(self.nb_agent):

                        experiences = self.memory.sample()
                        self.learn(agent_id, experiences, self.GAMMA)

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

    def learn(self, agent_id, experiences, gamma):
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
        
        states, _, _, next_states, _ = experiences

        #_whatis(states, texte="State Shape from experience [512][2][24]", block=True)
        with torch.no_grad():
            actions_next = torch.cat([self.agents[i].actor_target(extract_agent_data(next_states, i)) for i in range(self.nb_agent)], dim=1)

        for agent in self.agents:
            actions_pred = torch.cat([agent.actor_local(extract_agent_data(states, i)) if i == agent.id else agent.actor_local(extract_agent_data(states, i)).detach() for i in range(self.nb_agent)], dim=1)
        
            #_whatis(actions_next, texte="actions_next should be [512][4]", block=True)

            agent.learn(agent_id, experiences, gamma, actions_next, actions_pred)

        self.optimized = True


    def save(self, value):
        for agent in self.agents:
            agent.save(value)

    def load(self, value):
        for agent in self.agents:
            agent.load(value)