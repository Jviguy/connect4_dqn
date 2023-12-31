import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        priority = 0
        if reward < -40:
            priority = 10
        if reward < -190:
            priority = 20
        if reward > 20:
            priority = 5
        self.buffer.append((state, action, reward, next_state, done, priority))

    def sample(self, batch_size):
        # Normalize the priorities to create a probability distribution
        total_priority = sum([experience[5] for experience in self.buffer])  # Sum of all priorities
        probabilities = [experience[5] / total_priority for experience in self.buffer]

        # Sample experiences based on the defined probabilities
        sampled_indices = np.random.choice(range(len(self.buffer)), size=batch_size, p=probabilities)
        minibatch = [self.buffer[idx] for idx in sampled_indices]

        return minibatch

    def size(self):
        return len(self.buffer)
