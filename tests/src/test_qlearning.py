import unittest
from src import qlearning as module
import gym



class Testqlearning(unittest.TestCase):
    def setUp(self):
        self.sample_model = gym.make("Taxi-v2").env


