from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)
for i in range(5):
    print("Location {}".format(i))
    agent.show_Q(env, i, 0)
    agent.show_Q(env, i, 1)
    agent.show_Q(env, i, 2)
    agent.show_Q(env, i, 3)
