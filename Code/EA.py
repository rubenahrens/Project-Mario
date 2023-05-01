"""
Evolutionary Algorithm (EA) for the optimization of the parameters of the
network used to play super mario bros.
"""

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import tensorflow as tf

class EA():
    def __init__(self, config):
        self.config = config
        # self.num_params = self.env.observation_space.shape[0]
        # self.theta = np.zeros(self.num_params)

    def get_action(self, observation, theta=None):
        raise NotImplementedError
        # if theta is None:
        #     theta = self.theta
        # action = np.matmul(theta, observation)
        # action = np.tanh(action)
        # return action

    def get_reward(self, env, theta=None, render=False):
        raise NotImplementedError
        # observation = env.reset()
        # total_reward = 0
        # for t in range(self.num_episode):
        #     if render:
        #         env.render()
        #     action = self.get_action(observation, theta)
        #     observation, reward, done, info = env.step(action)
        #     total_reward += reward
        #     if done:
        #         break
        # return total_reward

    def train(self):
        raise NotImplementedError
        # for i in range(self.num_episode):
        #     N = np.random.randn(self.population_size, self.num_params)
        #     R = np.zeros(self.population_size)
        #     for j in range(self.population_size):
        #         theta_try = self.theta + self.sigma * N[j]
        #         R[j] = self.get_reward(self.env, theta_try)
        #     A = (R - np.mean(R)) / np.std(R)
        #     self.theta = self.theta + self.learning_rate / (self.population_size * self.sigma) * np.dot(N.T, A)
        #     self.sigma = self.sigma * self.decay
        #     reward = self.get_reward(self.env)
        #     print('Episode: ', i, 'Reward: ', reward, 'Sigma: ', self.sigma)
        
def main():
    env = gym.make('SuperMarioBros-v0')
    
    config = {
        "population_size":50,
        "sigma":0.1,
        "learning_rate":0.03,
        "decay":0.999,
        "num_episode":1000
    }
    
    EA = EA(config)
    
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