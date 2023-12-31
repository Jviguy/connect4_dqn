import math
import os.path

import numpy as np
from keras import Sequential, Input
from keras.src.layers import Dense, Flatten
from keras.src.optimizers import Adam
from keras import models

from replay_buffer import ReplayBuffer
import random


class DQNAgent:
    def __init__(self, state_size, action_size, name):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(1000000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        if os.path.isfile('models/'+name+".keras"):
            print("Loading saved models for agent:", name)
            self.model = models.load_model('./models/'+name+".keras")
            self.target_model = models.load_model('./models/'+name+"_target.keras")
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        self.name = name

    def _build_model(self):
        model = Sequential()
        model.add(Input(self.state_size))
        model.add(Flatten())
        model.add(Dense(264, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, env):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)[0]
        for i in range(len(act_values)):
            if env.height[i] < 0:
                act_values[i] = -math.inf
        return np.argmax(act_values)

    def act_target(self, state, env):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        act_values = self.target_model.predict(state, verbose=0)[0]
        for i in range(len(act_values)):
            if env.height[i] < 0:
                act_values[i] = -math.inf
        return np.argmax(act_values)

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)

        # Preparing the batch data
        states = np.array([experience[0] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])

        # Predicting Q-values for current states and next states
        current_qs_list = self.model.predict(states, verbose=0)
        future_qs_list = self.target_model.predict(next_states, verbose=0)

        # Initialize training data arrays
        x = np.zeros((batch_size, *self.state_size))
        y = np.zeros((batch_size, self.action_size))

        for index, (state, action, reward, next_state, done, priority) in enumerate(minibatch):
            if not done:
                max_future_q = np.amax(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # Append to training data
            x[index] = state
            y[index] = current_qs

        # Fit on all samples as one batch
        self.model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save('./models/'+self.name+".keras")
        self.target_model.save('./models/' + self.name + "_target.keras")