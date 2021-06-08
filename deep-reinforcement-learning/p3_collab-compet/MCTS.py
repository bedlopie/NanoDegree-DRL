import numpy as np


class MCTS():

    def __init__(self, state) -> None:
        
        self.state = state
        self.visited = 0
        self.value = 0
        self.expanded = False
        
        self.childern = []

