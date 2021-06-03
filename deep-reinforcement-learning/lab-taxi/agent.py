import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 1.0
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon_step = 0.99997
        self.epsilon_low = 0.00000001
        self.rng = np.random.default_rng()

    def epsilon_update(self):
        self.epsilon *= self.epsilon_step
        if self.epsilon < self.epsilon_low:
            self.epsilon = self.epsilon_low

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        probs = [self.epsilon/self.nA for i in range(self.nA)]
        probs[np.argmax(self.Q[state])] = 1.0 - self.epsilon + (self.epsilon / self.nA)
        action = self.rng.choice(np.arange(self.nA), p=probs)
        return action

    def expected_Q(self, actions):
        """ """
        probs = [self.epsilon/self.nA for i in range(self.nA)]
        probs[np.argmax(actions)] = 1.0 - self.epsilon + (self.epsilon / self.nA)
        action = np.dot(actions, probs)
        return np.sum(action)

    def get_equivalent_state(self, state, next_state):
        
        def encode(taxi_row, taxi_col, pass_loc, dest_idx):
            # (5) 5, 5, 4
            i = taxi_row
            i *= 5
            i += taxi_col
            i *= 5
            i += pass_loc
            i *= 4
            i += dest_idx
            return i

        def decode(i):
            out = []
            out.append(i % 4)
            i = i // 4
            out.append(i % 5)
            i = i // 5
            out.append(i % 5)
            i = i // 5
            out.append(i)
            assert 0 <= i < 5
            return reversed(out)

        states = []
        next_states = []
        
        taxi_row, taxi_col, pass_loc, _ = decode(state)
        if pass_loc != 4:
            for i in range(4):
                if i != pass_loc:
                    states.append(encode(taxi_row, taxi_col, pass_loc, i))
        else:
            states.append(state)        

        taxi_row, taxi_col, pass_loc, _ = decode(next_state)
        if pass_loc != 4:
            for i in range(4):
                if i != pass_loc:
                    next_states.append(encode(taxi_row, taxi_col, pass_loc, i))
        else:
            next_states.append(next_state)        

        if len(states) != len(next_states):
            next_states = [next_states[0]] * 4

        return zip(states, next_states)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        prev_Q = self.Q[state][action]
        next_action = np.argmax(self.Q[next_state])
        #next_action = self.select_action(next_state)

        if done:
            self.Q[state][action] = prev_Q + self.alpha * (reward - prev_Q)
        else:
            #print(" previous", state, next_state)
            for (state_i, next_state_i) in self.get_equivalent_state(state, next_state):
                #self.Q[state_i][action] = prev_Q + self.alpha * (reward + self.gamma * self.Q[next_state_i][next_action] - prev_Q)
                #print(state_i, next_state_i)
                self.Q[state_i][action] = self.Q[state_i][action] + self.alpha * (reward + self.gamma * self.Q[next_state_i][next_action] - self.Q[state_i][action])
                #self.Q[state][action] = prev_Q + self.alpha * (reward + self.gamma * self.expected_Q(self.Q[next_state]) - prev_Q)
            
        self.epsilon_update()

    def show_Q(self, env, loc, dest):
        mapping = ["↓", "↑", "→", "←", "P", "D"]
        walls =  [[" : ", " | ", " : ", " : ", " | " ],
                  [" : ", " | ", " : ", " : ", " | " ],
                  [" : ", " : ", " : ", " : ", " | " ],
                  [" | ", " : ", " | ", " : ", " | " ],
                  [" | ", " : ", " | ", " : ", " | " ] ]

        for i in range(5):
            print(" ")
            print(" ", end=" | ")
            for j in range(5):
                print(("      "+"{:.2f}".format(max(self.Q[env.encode(i, j, loc, dest)])))[-6:], end=walls[i][j])
            for j in range(5):
                print("{}".format(mapping[np.argmax(self.Q[env.encode(i, j, loc, dest)])]), end=walls[i][j])
        print(" ")