

import sys, time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import multiprocessing as mp
import warnings
import pickle
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import os

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

password = {
    '00': 1,
    '01': 2,
    '10': 3,
    '11': 4
}

def DECODE(sample: list) -> list:
    result = []
    for i in range(0, len(sample), 2):
        x = str(sample[i]) + str(sample[i+1])
        result.append(password[x])
    return result


def INDIVIDUAL_FITNESS(sample: list):
    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env.reset()
    actions = DECODE(sample)
    count = 0
    oldPos = 40
    for action in actions:
        state, reward, done, info = env.step(action)
        count += 1
        if count % 50 == 0:
            if oldPos == info['x_pos']:
                break
            else:
                oldPos = info['x_pos']
        if done:
            break
    env.close()
    fitness = info['x_pos']
    return fitness

def _INDIVIDUAL_FITNESS(sample: list):
    env = gym.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env.reset()
    actions = DECODE(sample)
    count = 0
    oldPos = 40
    for action in actions:
        state, reward, done, info = env.step(action)
        count += 1
        if count % 50 == 0:
            if oldPos == info['x_pos']:
                break
            else:
                oldPos = info['x_pos']
        if done:
            break
    env.close()
    fitness = info['x_pos']

    return fitness, count




def CAL_FITNESS(POPULATION: list) -> list:
    '''
    Calcualte the fitness of each individual of population
    Input:
      POPULATION: a list, each element represents the individual
    Output:
      fitness: a list, which has same length as POPULATION, the fitness values of POPULATION
    '''


    p = mp.Pool(processes=10)
    fitness = p.map(INDIVIDUAL_FITNESS, POPULATION)
    p.close()
    # for sample in POPULATION:
    #     fitness.append(INDIVIDUAL_FITNESS(sample))
    return fitness


def SELECTION(POPULATION: list, fitness: list) -> list:
    '''
    Based on the fitness values of population, generating a new population
    Input:
      POPULATION: a list, each element represents the individual
      fitness: a list, each element represents the fitness value of each individual in POPULATION
    Output:
      NEXT_POPLATION: a list, each element represents the individual
    '''
    fitness = np.array(fitness)
    overall_fitness = np.sum(fitness)
    choose_probability = fitness / overall_fitness

    NEXT_POPLATION = []
    M = len(POPULATION)
    while len(NEXT_POPLATION) != M:
        individualities = np.random.choice(M, p=choose_probability, size=(2,))
        if fitness[individualities[0]] > fitness[individualities[1]]:
            NEXT_POPLATION.append(POPULATION[individualities[0]])
        else:
            NEXT_POPLATION.append(POPULATION[individualities[1]])

    return NEXT_POPLATION


def CROSSOVER(POPULATION: list, Pc: float) -> list:
    '''
      Based on the cross-over probability, performs corss-over of population
      Input:
        POPULATION: a list, each element represents the individual
        Pc: cross-over probability
      Output:
        NEXT_POPLATION: a list, each element represents the individual
      '''
    M = len(POPULATION)
    NEXT_POPLATION = []

    for i in range(0, M, 2):
        father = POPULATION[i]
        mather = POPULATION[i + 1]
        if np.random.uniform() < Pc:
            crossover_point = np.random.choice(6000, 1)[0]
            son1 = np.empty(6000, dtype=int)
            son2 = np.empty(6000, dtype=int)
            son1[:crossover_point] = father[:crossover_point]
            son1[crossover_point:] = mather[crossover_point:]

            son2[:crossover_point] = mather[:crossover_point]
            son2[crossover_point:] = father[crossover_point:]
            NEXT_POPLATION.append(son1)
            NEXT_POPLATION.append(son2)
        else:
            NEXT_POPLATION.append(father)
            NEXT_POPLATION.append(mather)

    return NEXT_POPLATION

def PROTECT(POPULATION: list, index: int, PROTECT_LENGTH: int) -> list:
    NEW_POPULATION = []
    best = POPULATION[index]
    for sample in POPULATION:
        NEW_POPULATION.append(np.append(best[:PROTECT_LENGTH], sample[PROTECT_LENGTH:]))
    return NEW_POPULATION


def MUTATION(POPULATION: list, Pm: float) -> list:
    '''
      Based on the cross-over probability, performs mutation of population
      Input:
        POPULATION: a list, each element represents the individual
        Pm: mutation probability
      Output:
        NEXT_POPLATION: a list, each element represents the individual
    '''
    M = len(POPULATION)
    NEXT_POPLATION = []
    for i in range(M):
        x = np.random.choice([False, True], p=[1 - Pm, Pm], size=(6000,))

        z = np.array(POPULATION[i])
        z = z ^ x
        NEXT_POPLATION.append(z)

    return NEXT_POPLATION


def GA(M=10, T=500, Pc=0.7, Pm=0.3):
    # M = 10 # size of population
    # T = 500 # iterations
    # Pc = 0.7 # cross-over probability
    # Pm = 0.3 # mutation probability

    # generate population
    POPULATION = []  # population
    for i in range(M):
        x = np.random.choice([0, 1], size=(6000,))
        POPULATION.append(x)

    for iter in range(T):
        print("================= Generation: ", iter, " =================")
        # calculate fitness
        s = time.time()
        fitness = CAL_FITNESS(POPULATION)

        max_fitness = np.max(fitness)
        print("Maximum Fitness: ", max_fitness)
        print("Average Fitness: ", np.average(fitness))
        print('\n')

        # _, PROTECT_LENGTH = _INDIVIDUAL_FITNESS(POPULATION[np.argmax(fitness)])
        # PROTECT_LENGTH = PROTECT_LENGTH * 2 - 20
        # POPULATION = PROTECT(POPULATION, np.argmax(fitness), PROTECT_LENGTH)


        if max_fitness >= 3000:
            pickle.dump(POPULATION[np.argmax(fitness)], open("GARESULT.pkl", "wb"))
            break
        # selection
        POPULATION = SELECTION(POPULATION, fitness)
        # print("END SELECTION")
        # cross-over
        POPULATION = CROSSOVER(POPULATION, Pc)
        # print("END CROSSOVER")
        # mutation
        POPULATION = MUTATION(POPULATION, Pm)
        # print("END MUTATION")

        print("Time Consuming: ", time.time() - s)


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    # np.random.seed(20221204)
    start = time.time()
    GA(M=20, T=10000, Pc=0.7, Pm=0.5)
    end = time.time()
    print("The program takes %s seconds" % (end - start))
