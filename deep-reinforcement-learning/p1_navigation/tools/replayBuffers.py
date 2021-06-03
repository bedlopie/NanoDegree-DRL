import numpy as np
import torch
import random
from .binaryTreeSearch import sumTree, minTree, maxTree
from .hyperParameters import geometricParameter, linearParameter, Hyperparameters
from collections import namedtuple, deque

class preBuffer():

    def __init__(self, buffer_size, batch_size, device, alpha, TD_Error_clip, beta, seed):
        
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._device = device
        self._seed = seed

        self._rng = np.random.default_rng(self._seed)

        self._e = 0.001
        
        self.alpha = Hyperparameters(**alpha)
        self.TD_Error_clip = TD_Error_clip
        self.beta = Hyperparameters(**beta)

        self._experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._sumTree = sumTree(self._buffer_size)
        self._minTree = minTree(self._buffer_size)
        self._maxTree = maxTree(self._buffer_size)

    def add(self, state, action, reward, next_state, done):

        new_priority = self._maxTree.max()
        if new_priority == float('-inf'):
            new_priority = 1.
        new_experience = self._experience(state, action, reward, next_state, done)
        self._sumTree.add(new_priority, new_experience)
        self._minTree.add(new_priority, new_experience)
        self._maxTree.add(new_priority, new_experience)

    def sample(self):

        sum_value = self._sumTree.sum()
        #print(sum_value)
        boundaries = [i*(sum_value/float(self._batch_size)) for i in range(self._batch_size+1)]
        
        #print("\nBOUND", boundaries)
        tree_values = self._rng.uniform(boundaries[:-1], boundaries[1:])
        
        for i, v in enumerate(tree_values):
            if v > sum_value:
                print('OUT OF RANGE', i, v)

        sample_tree_indexes = [self._sumTree._sample_one(value, 0) for value in tree_values]
        sample_tree_values = [self._sumTree._tree[tree_index] for tree_index in sample_tree_indexes]
        sample_experiences = [self._sumTree._data[self._sumTree._data_index(tree_index)] for tree_index in sample_tree_indexes]

        states = torch.from_numpy(np.vstack([e.state for e in sample_experiences if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e.action for e in sample_experiences if e is not None])).long().to(self._device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sample_experiences if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in sample_experiences if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e.done for e in sample_experiences if e is not None]).astype(np.uint8)).float().to(self._device)

        # print priority and rewards
        #print("\nMini Batch min={} max={}".format(min(sample_tree_values), max(sample_tree_values)))
        #for i, p, r in zip(sample_tree_indexes, sample_tree_values, [e.reward for e in sample_experiences if e is not None]):
        #    print(i, p,r)
        #print("End of Mini Batch")

        # Calculate priorities
        min_value = self._minTree.min()
        #min_value = min(sample_tree_values)
        weights = [(min_value / sample_tree_value)**self.beta.value for sample_tree_value in sample_tree_values]
        #print("\nWEIGTHS", min_value, weights)

        if max(weights) > 1:
            print("Error")
            print(min_value)
            print(sample_tree_values)
            print(weights)
            print(boundaries)
            print(tree_values)
            print(self._sumTree._sample_one(tree_values[-1], 0))
            print("Fin Error")

        #small reshape to numpy
        sample_tree_indexes = np.array(sample_tree_indexes).astype(np.int64)
        weights = torch.from_numpy(np.array(weights)).float().unsqueeze(1).to(self._device)

        # _ = input("Continue ? ")

        return (states, actions, rewards, next_states, dones, sample_tree_indexes, weights)

    def update(self, indexes, TD_Error):

        TD_Error = np.clip(torch.abs(TD_Error.detach()).cpu().numpy(), *self.TD_Error_clip)
        new_priorities = np.power(TD_Error+self._e, self.alpha.value)

        if min(new_priorities) == 0.0:
            print("Pas Normal", TD_Error, new_priorities)

        for index, new_priority in zip(indexes, new_priorities):
            self._sumTree.update(index, new_priority)
            self._minTree.update(index, new_priority)
            self._maxTree.update(index, new_priority)

    def statistics(self):
        rewards = [e.reward for e in self._sumTree._data if e is not None]
        count_0 = sum([1 for r in rewards if r==0])
        count_1 = sum([1 for r in rewards if r==1])
        count_minus1 = sum([1 for r in rewards if r==-1])
        return "\tStatistics (L/D/W/Sum) {} / {} / {} / {:.1f}".format(count_minus1, count_0, count_1, self._sumTree.sum())

    def __len__(self):
        return len(self._sumTree)


class replayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.memory = deque(maxlen=buffer_size) 
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._device = device
        self.seed = seed
        self._rng = np.random.default_rng(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self._device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self._device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self._device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self._device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self._device)
  
        return (states, actions, rewards, next_states, dones)

    def statistics(self):
        rewards = [e.reward for e in self.memory if e is not None]
        count_0 = sum([1 for r in rewards if r==0])
        count_1 = sum([1 for r in rewards if r==1])
        count_minus1 = sum([1 for r in rewards if r==-1])
        return "\tStatistics {} / {} / {} (L/D/W)".format(count_minus1, count_0, count_1)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
