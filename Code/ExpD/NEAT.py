import sys
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import os
import neat
import pickle
import multiprocessing as mp
import cv2
import time
import argparse



ACTIONDICT = {
    'simple': SIMPLE_MOVEMENT,
    'right': RIGHT_ONLY
}

class NEAT():
    def __init__(self, generations, parallel=10, rewardasFitness=False, scaleFactor=20, filefolder='neat', actionSet='simple'):
        self.generations = generations
        self.thredNum = parallel
        self.filefolder = filefolder
        if not os.path.exists(self.filefolder):
            os.mkdir(self.filefolder)


        ## image scale factor
        scaleFactor = scaleFactor
        env = gym.make('SuperMarioBros-1-1-v1')
        width, length, _ = env.observation_space.shape
        self.width = int(width / scaleFactor)
        self.length = int(length / scaleFactor)

        self.rewardasFitness = rewardasFitness
        self.actionSet = actionSet


    def get_action(self, output):
        return output.index(max(output))

    def process_state(self, state, info):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (self.width, self.length))
        state = state.flatten()
        state = np.append(state, int(info['x_pos']))
        state = np.append(state, int(info['x_pos_screen']))
        state = np.append(state, int(info['y_pos']))
        return state

    def cal_fitness(self, para):
        genome = para[0]
        config = para[1]

        env = gym.make('SuperMarioBros-1-1-v1')
        if self.actionSet == 'full':
            pass
        else:
            env = JoypadSpace(env, ACTIONDICT[self.actionSet])

        obs = env.reset()
        obs, reward, done, info = env.step(1)
        ## NEAT: we can choose different network type, RNN or simple feed forward network
        # net = neat.nn.FeedForwardNetwork.create(genome, config)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        done = False


        count = 0
        oldPos = 0
        fitness = 0
        overallReward = 0
        while not done:
            state = self.process_state(obs, info)
            output = self.get_action(net.activate(state))
            obs, reward, done, info = env.step(output)
            overallReward += reward
            count += 1
            if count % 50 == 0:
                if oldPos == info['x_pos']:
                    break
                else:
                    oldPos = info['x_pos']


        fitness = int(info['x_pos'])
        if self.rewardasFitness:
            fitness = overallReward


        if info['flag_get']:
            fitness = 3500

        env.close()
        return fitness


    def eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        p = mp.Pool(processes=self.thredNum)
        data = p.map(self.cal_fitness, [(genome, config) for genome in genomes])
        p.close()

        for i in range(len(genomes)):
            genomes[i].fitness = data[i]

    def run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self.eval_genomes, n)
        pickle.dump(stats, open(self.filefolder + '/stats.pkl', 'wb'))
        pickle.dump(winner, open(self.filefolder + '/winner.pkl', 'wb'))

    def experiment(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self.run(config_path, self.generations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, help='Path of NEAT configuration file', default='config')
    parser.add_argument('--p', type=int, help='Number of thread for parallel running', default=10)
    parser.add_argument('--r', action='store_true', help='Use reward as fitness')
    parser.add_argument('--f', type=int, help='Scale factor for downsample the image', default=10)
    parser.add_argument('--z', type=str, help='Path of the result', default='config')
    parser.add_argument('--a', type=str, help='Action Sample Set', default='simple')
    args = parser.parse_args()

    s = time.time()
    t = NEAT(generations=1000, parallel=args.p, rewardasFitness=args.r, scaleFactor=args.f, filefolder=args.z, actionSet='simple')
    t.experiment(args.c)
    print("Running Time: ", time.time() - s)