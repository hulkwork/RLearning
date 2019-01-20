# INITIALIZATION: libraries, parameters, network...

from keras.models import Sequential  # One layer after the other
from keras.layers import Dense, Flatten, \
    Conv2D  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque  # For storing moves

import numpy as np
import random  # For sampling batches from the observations


class DeepQLearningPixel(object):
    def __init__(self, n_state, n_actions, input_shape, memory_size=50):
        self.n_state = n_state
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.model = self._build_model()
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.batch_size = 12

    def _build_model(self):
        # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(2, 2), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(8, kernel_size=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.n_actions, kernel_initializer='uniform',
                        activation='sigmoid'))  # Same number of outputs as possible actions
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def memories(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_actions)
        act_values = self.model.predict(np.array([state]))
        action = np.argmax(act_values[0])
        return action

    def exploration(self, env, epochs=10, observetime=100):
        for e in range(epochs):
            env.reset()
            # state = env.reset()
            state_image = env.render("rgb_array")
            #env.render()
            for time in range(observetime):
                #
                action = self.act(state_image)
                next_state, reward, done, _ = env.step(action)
                print(reward)
                reward = float(reward) if not done else -10.0
                # next_state = np.reshape(next_state, [1, state_size])
                next_state_image = env.render("rgb_array")
                env.render("human")

                self.memories(state_image, action, reward, next_state_image, done)
                # state = next_state
                state_image = next_state_image

                if done:
                    print("episode: {}/{}, score: {}, time {}, e: {:.2}"
                          .format(e, epochs, reward, time, self.epsilon))
                    state_image = env.render("rgb_array")
                    break
                if len(self.memory) > self.batch_size:
                    self.replay()
                print("episode: {}/{}, score: {}, time {}, e: {:.2}, done {}"
                          .format(e, epochs, reward, time, self.epsilon, done))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        targets = np.zeros((self.batch_size, self.n_actions))
        inputs = []
        for i, (state_image, action, reward, next_state_image, done) in enumerate(minibatch):
            targets[i] = self.model.predict(np.array([state_image]))[0]
            Q_sa = self.model.predict(np.array([next_state_image]))[0]
            inputs.append(state_image)

            targets[i, action] = reward
            if not done:
                targets[i, action] = reward + self.gamma * np.max(Q_sa)

        self.model.train_on_batch(np.array(inputs), targets)

    def play(self, env):
        env.reset()
        state = env.render("rgb_array")
        done = False
        tot_reward = 0.0
        while not done:
            env.render()  # Uncomment to see game running
            Q = self.model.predict(np.array([state]))[0]
            action = np.argmax(Q)
            observation, reward, done, info = env.step(action)
            state = env.render("rgb_array")
            tot_reward += reward
        print('Game ended! Total reward: {}'.format(tot_reward))



