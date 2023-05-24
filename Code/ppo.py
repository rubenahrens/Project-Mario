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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import datetime
import os

# LOG_DIR = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

models_names = os.listdir('models/')
config = {
    "population_size":50,
    "sigma":0.1,
    "learning_rate":0.000001,
    "decay":0.999,
    "num_episode":1000,
    "stages":['1-2']
}

def build_env(level=1):
    # 1. Create the base environment
    env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v3', stages=config["stages"])
    # 2. Simplify the controls 
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 3. Grayscale
    env = GrayScaleObservation(env, keep_dim=True)
    # 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
    # 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')
    return env

def train_model(model):
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save('models/'+models_names[-1][:-5]+str(int(models_names[-1][-5])+1))

def main():
    
    
    env = build_env()
    if input("load? (y/n): ") == 'y':
        model = PPO.load('models/'+models_names[int(input(f"which model? (index of {models_names}): "))][:-4], env=env, verbose=0,
                         learning_rate=config["learning_rate"],n_steps=512, device='cuda')
        if input("train? (y/n): ") == 'y':
            model = train_model(model)
        else:
            state = env.reset()
            rewards = []
            total_reward = 0
            while True: 

                action, _ = model.predict(state)
                state, reward, done, [info] = env.step(action)
                if done or info['flag_get']:
                    rewards.append(total_reward)
                    total_reward = 0
                    env.reset()
                else:
                    total_reward += reward

                if len(rewards) > 100:
                    break

            rewards = np.array(rewards)

            print(rewards.min(), rewards.mean(), rewards.max())
    else:
        model = PPO('CnnPolicy', env, verbose=0, learning_rate=config["learning_rate"], 
            n_steps=512, device='cuda')
        train_model(model)

if __name__ == "__main__":
    main()