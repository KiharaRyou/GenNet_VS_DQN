import random
import numpy as np

# Initialize the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, size):
        return np.array(random.sample(self.buffer, size))

class DQNAgent:
    def __init__(self, model, action_size, replay_buffer_size=5000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.memory = ReplayBuffer(replay_buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_network = model

    def act(self, state):
        # choose an action using the epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, terminated, truncated):
        experience = (state, action, reward, next_state, terminated, truncated)
        self.memory.add(experience)

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return
        sample_batch = random.sample(self.memory.buffer, batch_size)
        for state, action, reward, next_state, terminated, truncated in sample_batch:
            target = reward
            if not terminated:
              target = reward + self.gamma * np.amax(self.q_network.predict(next_state)[0])
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    # def save(self, name):
