import sys
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
import os
import neat
import pickle
import multiprocessing as mp
import visualize
import cv2
import warnings
import time
warnings.filterwarnings("ignore")


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



class EA():
    def __init__(self, generations, parallel=2):
        # self.config = config
        self.generations = generations
        self.par = parallel

        IMG_SCALE_FACTOR = 10
        env = gym.make('SuperMarioBros-1-1-v1')
        width, length, channels = env.observation_space.shape
        self.width = int(width / IMG_SCALE_FACTOR)
        self.length = int(length / IMG_SCALE_FACTOR)


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
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=2)
        # env.configure(lock=self.lock)

        obs = env.reset()
        obs, reward, done, info = env.step(1)
        ## NEAT: we can choose different network type, RNN or simple feed forward network
        # net = neat.nn.FeedForwardNetwork.create(genome, config)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        done = False

        count = 0
        oldPos = 40
        while not done:
            state = self.process_state(obs, info)
            output = self.get_action(net.activate(state))
            obs, reward, done, info = env.step(output)
            count += 1
            if count % 50 == 0:
                if oldPos == info['x_pos']:
                    break
                else:
                    oldPos = info['x_pos']

        fitness = int(info['x_pos'])

        if info['flag_get']:
            fitness = 100000000

        if fitness >= 3000:
            ## just for test
            pickle.dump(genome, open("finisher.pkl", "wb"))
        env.close()

        return fitness


    def eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        p = mp.Pool(processes=self.par)
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
        p.add_reporter(neat.Checkpointer(5))

        print("loaded checkpoint...")
        winner = p.run(self.eval_genomes, n)
        win = p.best_genome
        pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open('real_winner.pkl', 'wb'))

        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self.run(config_path, self.generations)


if __name__ == "__main__":
    s = time.time()
    t = EA(10000, parallel=10)
    t.main()
    print("Running Time: ", time.time() - s)