import numpy as np
import random
random.seed(2019)


class NoImplemented(Exception): pass


class QLearning(object):
    def __init__(self, n_actions, n_states):
        """
        Define q_table

        :param n_actions: number of actions
        :param n_states: number of states
        """
        self.n_actions = n_actions
        self.n_states = n_states
        self.q_table = np.zeros((self.n_states, self.n_actions))
        print(self.q_table.shape)

    def train(self, model, iteration, alpha=0.1, gamma=0.6, epsilon=0.1, best_n_epoch=12):
        """

        :param model: from gym api
        :param iteration: number of training iterations
        :param alpha: learning rate
        :param gamma: discount next reward of a action
        :param epsilon: random threshold for action
        :return:
        """


        all_epochs = []
        for i in range(1, iteration):
            state = model.reset()

            # Init Vars
            epochs, penalties, reward, = 0, 0, 0
            done = False
            r = 0

            while not done:
                if random.uniform(0, 1) < epsilon:
                    # Check the action space
                    action = model.action_space.sample()
                else:
                    # Check the learned values
                    action = np.argmax(self.q_table[state])

                next_state, reward, done, info = model.step(action)
                r += reward

                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])

                # Update the new value
                new_value = (1 - alpha) * old_value + alpha * \
                            (reward + gamma * next_max)
                self.q_table[state, action] = new_value

                state = next_state
                epochs += 1
                if done and epochs <=best_n_epoch :
                    print("Done after %d epochs and %.2f at %d iteration" % (epochs, r, i))

            if i % 100 == 0:
                print("Episode: %d" % i)

    def best_value(self, model):
        state = model.reset()
        done = False
        n_step = 0
        steps = []
        total_rewards = 0
        while not done:
            action = np.argmax(self.q_table[state])
            steps.append(action)
            next_state, reward, done, info = model.step(action)
            total_rewards += reward
            state = next_state
            n_step += 1
        print("Finished into %d step" % n_step)
        model.render(mode='rgb_array')
        return steps
