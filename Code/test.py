import sys

import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
# import tensorflow as tf
import os
import neat
import pickle
import multiprocessing as mp
import visualize
import cv2
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
#
# env = gym.make('SuperMarioBros-1-1-v0')
# env = JoypadSpace(env, RIGHT_ONLY)
#
# print(env.get_action_meanings())
#
# state = env.reset()
# for i in range(1000):
#     if i < 150:
#         action = 1
#     else:
#         try:
#             action = int(input("Please input action (0-5): "))
#         except:
#             action = int(input("Please input action (0-5): "))
#     # if i % 2 == 0:
#     #     state, reward, done, info = env.step(1)
#     #
#     # if i % 2 != 0:
#     #     state, reward, done, info = env.step(2)
#     state, reward, done, info = env.step(action)
#     if done:
#         break
#     env.render()
# env.close()
# # state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
# # state = cv2.pyrDown(state)



#### CODE BELOW DEMOSTRATE THAT THE MULTIPROCESSING PART OF VIVI CODE IS WRONG ####

# from multiprocessing import Pool
# import multiprocessing as mp
#
# def job(num):
#     return num * 2
#
# def job_N(num, o):
#     o.put(num * 2)
#
# if __name__ == '__main__':
#     p = Pool(processes=3)
#     data = p.map(job, [i for i in range(50)])
#     p.close()
#     print(data)
#
#     output = mp.Queue()
#     processes = [mp.Process(target=job_N, args=(i, output)) for i in range(50)]
#
#     [p.start() for p in processes]
#     [p.join() for p in processes]
#
#     results = [output.get() for p in processes]
#     print(results)



