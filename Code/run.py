import neat
import pickle

import sys

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.wrappers import Monitor
import gym
import numpy as np
# import tensorflow as tf
import os
import neat
import pickle
import multiprocessing as mp
import visualize

import warnings
warnings.filterwarnings("ignore")

import visualize
import gzip
import neat.genome

from matplotlib import animation
import matplotlib.pyplot as plt
import cv2


class SkipFrame(gym.Wrapper):
    '''
        Custom wrapper that inherits from gy.Wrapper and implements the step() function.
        Use it to return only every skip nth frame
    '''

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

IMG_SCALE_FACTOR = 10
env = gym.make('SuperMarioBros-1-1-v1')
inx, iny, inc = env.observation_space.shape
inx = int(inx / IMG_SCALE_FACTOR)
iny = int(iny / IMG_SCALE_FACTOR)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


ACTIONS = [
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1],
]
# FILENAME = "./Files/gen_2284"
CONFIG = 'config'


def process_state(state, info):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (inx, iny))
    state = state.flatten()
    state = np.append(state, int(info['x_pos']))
    state = np.append(state, int(info['x_pos_screen']))
    state = np.append(state, int(info['y_pos']))
    return state


def main(config_file, file, level="1-1"):
    # with gzip.open(FILENAME) as f:
    #   config = pickle.load(f)[1]
    # print(str(config.genome_type.size))
    frames = []
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    genome = pickle.load(open(file, 'rb'))
    env = Monitor(gym.make('SuperMarioBros-1-1-v1'),  './video', force=True)
    # env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=2)

    env_x = Monitor(gym.make('SuperMarioBros-1-1-v0'), './video', force=True)
    # env = gym.make('SuperMarioBros-1-1-v0')
    env_x = JoypadSpace(env_x, SIMPLE_MOVEMENT)
    env_x = SkipFrame(env_x, skip=2)


    net = neat.nn.FeedForwardNetwork.create(genome, config)
    info = {'x_pos': 0}
    count = 0
    try:
        # while info['x_pos'] != 3252:
        while count != 1:
            state = env.reset()
            obs, reward, done, info = env.step(1)
            env_x.reset()
            env_x.step(1)
            done = False
            i = 0
            old = 40
            while not done:

                state = process_state(state, info)

                output = net.activate(state)
                ind = output.index(max(output))
                s, reward, done, info = env.step(ind)
                _, _, _, _ = env_x.step(ind)
                frames.append(env_x.render(mode="rgb_array"))
                state = s
                i += 1
                print(old, info['x_pos'])
                # if i % 50 == 0:
                #     if old == info['x_pos']:
                #         break
                #
                #     else:
                #         old = info['x_pos']
            print("Distance: {}".format(info['x_pos']))
            count += 1
        env.close()

        save_frames_as_gif(frames)
    except KeyboardInterrupt:
        env.close()
        exit()


if __name__ == "__main__":
    main(CONFIG, "winner.pkl")