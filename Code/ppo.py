"""
PPO algorithm

We will use the PPO algorithm to train the agent to play the game.
We will use the stable baselines library to implement the PPO algorithm.
https://www.youtube.com/watch?v=5P7I-xPq8u8
https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

"""

from gym.wrappers import GrayScaleObservation
import gym
import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import datetime

LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def build_env(level=1):
    # 1. Create the base environment
    env = gym.make('SuperMarioBrosRandomStages-v0', stages=['1-4', '2-4', '3-4', '4-4'])
    # 2. Simplify the controls 
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3. Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')
    return env

def train_model(model, env, config):
    model.learn(total_timesteps=10, progress_bar=True, tb_log_name="first_run")
    model.save('thisisatestmodel')

def main():
    
    config = {
        "population_size":50,
        "sigma":0.1,
        "learning_rate":0.000001,
        "decay":0.999,
        "num_episode":1000
    }
    
    env = build_env()
    if input("load? (y/n): ") == 'y':
        model = PPO.load('thisisatestmodel', env=env, verbose=0, tensorboard_log=LOG_DIR, learning_rate=config["learning_rate"],
                n_steps=512, device='cuda')
        if input("train? (y/n): ") == 'y':
            model = train_model(model, env, config)
    else:
        model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR, learning_rate=config["learning_rate"], 
            n_steps=512, device='cuda') 
        
    # Start the game 
    state = env.reset()
    # Loop through the game
    while True: 
        
        action, _ = model.predict(state)
        state, _, _, _ = env.step(action)
        env.render()

if __name__ == "__main__":
    main()