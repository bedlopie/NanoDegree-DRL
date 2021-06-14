import numpy as np





class MCTS_Node():

    def __init__(self, state, parent=None) -> None:
        
        self.state = state
        self.visited = 0
        self.value = 0
        self.expanded = False
        
        self.parent = parent
        self.possible_actions = []
        self.childern = []

    def evaluation(self):

        return 0.1

    def backpropagate(self, value):

        self.visited += 1
        self.value += value
        while (self.parent != None):
            self.parent.backpropagate(value)

    def add_child(self, action_index):

        pass


class MCTS():

    def __init__(self, environement) -> None:

        self.root = MCTS_Node
        self.environement = environement


    def set_root(self, node):
        self.root = node

    def search(self, type="Iteration", limit=100):

        if type == "Iteration" :
            print("Iteration limit =", limit)

            for _ in range(limit):
                node = self.find_leaf()
                eval = node.evaluate(self.environement)
                node.backgropagate(eval)

        else:
            print("Time limit =", limit)

            for _ in range(limit):
                node = self.find_leaf()
                eval = node.evaluate()
                node.backgropagate(eval)
    
    def find_leaf(self):

        node = self.root
        while node.terminal != True:
            if node.expanded == True:
                node.select_higher_ucbt()
            else:
