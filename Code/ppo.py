"""
PPO algorithm

We will use the PPO algorithm to train the agent to play the game.
We will use the stable baselines library to implement the PPO algorithm.
https://www.youtube.com/watch?v=5P7I-xPq8u8
https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

"""

import gym
import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO

# class PPO():
#     def __init__(self, config):
#     self.config = config

#     def get_action(self, observation, theta=None):
#         raise NotImplementedError

#     def get_reward(self, env, theta=None, render=False):
#         raise NotImplementedError

#     def train(self):
#         raise NotImplementedError

def main():
    env = gym.make('SuperMarioBros-v0')
    
    config = {
        "population_size":50,
        "sigma":0.1,
        "learning_rate":0.000001,
        "decay":0.999,
        "num_episode":1000
    }
    
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=config["learning_rate"], 
            n_steps=512) 
    
    model.learn(total_timesteps=1000000)
    
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation.shape)
        print(reward)
        print(done)
        print(info)


if __name__ == "__main__":
    main()