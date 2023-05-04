"""
Evolutionary Algorithm (EA) for the optimization of the parameters of the
network used to play super mario bros.
"""

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import os
import tensorflow as tf
import config
from algorithm import RANDOM as MODEL

def main(config):
    
    # RUNNING SETTINGS
    RENDER = config.run['render'] 
    PRINT = config.run['print'] 
    VERSION = config.run['version']
    ENV = gym.make(VERSION)
    TRAIN_PARAM = config.train
    
    Model = MODEL(TRAIN_PARAM, ENV, RENDER, PRINT)
    Model.train()
        
if __name__ == "__main__":
    try:
        main(config)
    except KeyboardInterrupt:
        pass
        
    