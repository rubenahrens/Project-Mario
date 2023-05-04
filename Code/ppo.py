"""
PPO algorithm

We will use the PPO algorithm to train the agent to play the game.
We will use the stable baselines library to implement the PPO algorithm.
https://www.youtube.com/watch?v=5P7I-xPq8u8
https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

"""

class PPO():
    def __init__(self, config):
    self.config = config

    def get_action(self, observation, theta=None):
        raise NotImplementedError

    def get_reward(self, env, theta=None, render=False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

if __name__ == "__main__":
    PPO()