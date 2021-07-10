import numpy as np
import torch
import random
import copy
from collections import namedtuple, deque

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.6):
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
        #_info(self.state)
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
    
    def add_experience(self, experience):
        """Add a new experience to memory."""
        self.memory.append(experience)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        Returns a tuple constituted of [N, *]
        * being whatever the format that variable was in.

        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([np.expand_dims(e.state, axis=0) for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([np.expand_dims(e.action, axis=0) for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([np.expand_dims(e.reward, axis=0) for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([np.expand_dims(e.next_state, axis=0) for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([np.expand_dims(e.done, axis=0) for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def stats(self):
        return_dict = {}
        for e in self.memory:
            #key = str(round(e.reward,2))
            key = round(e.reward,2)
            return_dict[key] = return_dict.get(key, 0) + 1

        return return_dict

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


    
class RewardBuffer:

    def __init__(self, buffer_size, gamma, device, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.gamma = [0.] + [gamma**i for i in range(buffer_size-1, 0, -1)]
        self.device = device
        self.seed = seed
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):

        experience = self.experience(state, action, reward, next_state, done)

        if len(self.memory) == 0:
            for _ in range(self.buffer_size):
                self.memory.append(experience)
            return experience
        else:
            for g, e in zip(self.gamma, self.memory):
                new_reward = e.reward + [r * g for r in reward]
                e = e._replace(reward=new_reward)

            last_experience = self.memory[0]
            self.memory.append(experience)    

            return last_experience

    def reset(self):
        self.memory.clear()

    def purge(self):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def _debug(msg):
    print("[DEBUG] ", *msg, sep=" | ")

def _assert(txt, bool_eq):
    if (bool_eq):
        print("[ASSERT] ", txt, " => True")
    else:
        print("[ASSERT] ", txt, " => False")

def _whatis(obj, texte="", block=False):

    if isinstance(obj, np.ndarray):
        print("[INFO] [{}] Numpy Array = ".format(texte), obj.shape)
    elif isinstance(obj, list):
        print("[INFO] [{}] List = ".format(texte), len(obj))
    elif isinstance(obj, torch.Tensor):
        print("[INFO] [{}] Torch tensor = ".format(texte), obj.shape)      
    else:
        print("[INFO] Error unknown type ", type(obj))
    
    if (block):
        _ = input()

def _info(obj, texte="", block=False):
    _whatis(obj, texte=texte, block=False)
    print(obj)
    print("[INFO] END")

    if (block):
        _ = input()


