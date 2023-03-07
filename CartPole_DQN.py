import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import pandas as pd
from DQN import DQNAgent
from create_model import create_network, create_dense_layer
import psutil
import time

start = time.time()
pid = psutil.Process().pid

env = gym.make('CartPole-v1')

batch_size = 32
episodes = 10
done = False

input_shape = env.observation_space.shape
action_size = env.action_space.n

dense_layer = create_dense_layer(32)
model = create_network(input_shape, action_size, [dense_layer])

agent = DQNAgent(model, action_size)

episode_col = []
score_col =[]

for episode in range(episodes):
    state, info = env.reset()
    state = np.reshape(state, [1, input_shape[0]])
    terminated = False
    score = 0
    while not terminated:
        # env.render()
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.reshape(next_state, [1, input_shape[0]])
        agent.remember(state, action, reward, next_state, terminated, truncated)
        state = next_state
        score += reward
    print("Episode {}# Score: {}".format(episode, score))
    agent.replay(batch_size)
    episode_col.append(episode)
    score_col.append(score)
df = pd.DataFrame({"episode": episode_col, "score": score_col}) 
df.to_csv('cartpole_dqn_result.csv')

end = time.time()
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")

memory_info = psutil.Process(pid).memory_info()
memory_usage = memory_info.rss / 1024.0 / 1024.0 # in MB
print(f"Total memory consumed: {memory_usage:.2f} MB")


    

