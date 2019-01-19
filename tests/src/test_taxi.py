from src.qlearning import QLearning
import gym

env = gym.make("Taxi-v2").env
env.seed(2019)

q_table = QLearning(n_states=env.observation_space.n, n_actions=env.action_space.n)
actions = [i for i in range(env.action_space.n)]


q_table.train(model=env, iteration=10000)
best_way = q_table.best_value(env)
print(best_way)

env = gym.make('CartPole-v0')
env.seed(2019)

# q_table = QLearning(n_states=env.observation_space, n_actions=env.action_space.n)
# actions = [i for i in range(env.action_space.n)]
#
#
# q_table.train(model=env, iteration=10000)
# best_way = q_table.best_value(env)
# print(best_way)


