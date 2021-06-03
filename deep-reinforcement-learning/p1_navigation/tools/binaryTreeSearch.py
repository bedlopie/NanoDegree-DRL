import numpy as np


class Tree():

    def __init__(self, buffer_size, operateur, neutre):

        def _upper_power_of_two(n):
            if (n and not(n & (n - 1))):
                return n
            p = 1    
            while (p < n):
                p <<= 1
            return p

        self._buffer_size = buffer_size
        self._operateur = operateur
        self._neutre = neutre

        self._memory_size = _upper_power_of_two(buffer_size)
        self._tree = np.array([neutre for _ in range(2*self._memory_size - 1)], dtype=np.float32)
        self._data = np.empty(self._memory_size, dtype=object)
        self._current_data_index = 0
        self._full = False

    def _tree_index(self, data_index):

        return (data_index + self._memory_size - 1)

    def _data_index(self, tree_index):

        return (tree_index - self._memory_size + 1)

    def add(self, value, buffered_object):

        tree_index = self._tree_index(self._current_data_index)
        self._data[self._current_data_index] = buffered_object
        self.update(tree_index, value)

        # self._current_data_index = (self._current_data_index + 1) % self._buffer_size
        if self._current_data_index == self._buffer_size-1:
            self._current_data_index = 0
            self._full = True
        else:
            self._current_data_index = (self._current_data_index + 1)

    def update(self, index, new_value):

        self._tree[index] = new_value
        self._update_up(index)

    def _update_up(self, index):

        other_branch_index = (index + 1) if index % 2 == 1 else (index - 1)
        parent_index = (index - 1) // 2

        self._tree[parent_index] = self._operateur(self._tree[index], self._tree[other_branch_index])

        if parent_index != 0:
            self._update_up(parent_index)

    def __len__(self):
        if self._full:
            return self._buffer_size
        else:
            return self._current_data_index


class sumTree(Tree):

    def __init__(self, buffer_size):
        super(sumTree, self).__init__(buffer_size, lambda x, y: x+y, 0.)

    def sum(self):
        return self._tree[0]

    def _sample_one(self, value, index):

        left_child_index = index * 2 + 1
        right_child_index = left_child_index + 1

        if right_child_index > (2*self._memory_size-1):
            return index

        if value > self._tree[left_child_index]:
            return self._sample_one(value - self._tree[left_child_index], right_child_index)
        else:
            return self._sample_one(value, left_child_index)


class minTree(Tree):

    def __init__(self, buffer_size):
        super(minTree, self).__init__(buffer_size, lambda x, y: min(x,y), float('inf'))

    def min(self):
        return self._tree[0]

class maxTree(Tree):

    def __init__(self, buffer_size):
        super(maxTree, self).__init__(buffer_size, lambda x, y: max(x,y), float('-inf'))

    def max(self):
        return self._tree[0]