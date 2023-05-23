"""
PPO algorithm

We will use the PPO algorithm to train the agent to play the game.
We will use the stable baselines library to implement the PPO algorithm.
https://www.youtube.com/watch?v=5P7I-xPq8u8
https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

"""

from gym.wrappers import GrayScaleObservation
import gym_super_mario_bros
import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import datetime

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def main():
    # 1. Create the base environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # 2. Simplify the controls 
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3. Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')
    
    config = {
        "population_size":50,
        "sigma":0.1,
        "learning_rate":0.000001,
        "decay":0.999,
        "num_episode":1000
    }
    
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=config["learning_rate"], 
            n_steps=512) 
    
    # train the agent until ctrl+c is pressed
    try:
        model.learn(total_timesteps=1000000)
    except KeyboardInterrupt:
        model.save('thisisatestmodel')
        print("Saved model")
        exit()
    
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