"""
Evolutionary Algorithm (EA) for the optimization of the parameters of the
network used to play super mario bros.
"""
import sys

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym
import numpy as np
# import tensorflow as tf
import neat
import os
import neat
import pickle
import multiprocessing as mp
import visualize

import warnings
warnings.filterwarnings("ignore")


class EA():
    def __init__(self, generations, parallel=2):
        # self.config = config
        self.generations = generations
        self.par = parallel

    def get_action(self, output):
        return output.index(max(output)) + 1

    def fitness(self, genomes, config):
        env = gym.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        idx, genomes = zip(*genomes)
        for genome in genomes:
            try:
                state = env.reset()
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                done = False
                i = 0
                # old = 40
                while not done:
                    state = state.flatten()
                    output = net.activate(state)
                    output = self.get_action(output)
                    s, reward, done, info = env.step(output)
                    state = s
                    i += 1
                    # if i % 50 == 0:
                    #     if old == info['distance']:
                    #         break
                    #     else:
                    #         old = info['distance']

                # [print(str(i) + " : " + str(info[i]), end=" ") for i in info.keys()]
                # print("\n******************************")

                fitness = info['x_pos']
                genome.fitness = fitness
                env.close()
            except KeyboardInterrupt:
                env.close()
                exit()

    def _fitness_func(self, genome, config, o):
        env = gym.make('SuperMarioBros-1-1-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        # env.configure(lock=self.lock)
        try:
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            i = 0
            old = 40
            while not done:
                state = state[:, :, 0]
                state = state.flatten()
                output = net.activate(state)
                # print(output)
                output = self.get_action(output)
                s, reward, done, info = env.step(output)
                fitness = int(info['x_pos'])
                # print("Step: {}, Action: {}, Reward: {}, Info: {}".format(i, output, reward, info))
                state = s
                i += 1

            # [print(str(i) + " : " + str(info[i]), end=" ") for i in info.keys()]
            # print("\n******************************")

            # fitness =  info['x_pos']
            print("Fitness: ", int(fitness))
            if fitness >= 3252:
                pickle.dump(genome, open("finisher.pkl", "wb"))
                env.close()
                print("Done")
                exit()
            o.put(fitness)
            env.close()
        except KeyboardInterrupt:
            env.close()
            exit()

    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)

        for i in range(0, len(genomes), self.par):
            output = mp.Queue()

            processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in
                         genomes[i:i + self.par]]

            [p.start() for p in processes]
            [p.join() for p in processes]

            results = [output.get() for p in processes]

            for n, r in enumerate(results):
                genomes[i + n].fitness = r

    def _run(self, config_file, n):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(neat.Checkpointer(5))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        print("loaded checkpoint...")
        winner = p.run(self._eval_genomes, n)
        win = p.best_genome
        pickle.dump(winner, open('winner.pkl', 'wb'))
        pickle.dump(win, open('real_winner.pkl', 'wb'))

        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

    def main(self, config_file='config'):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self._run(config_path, self.generations)

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
        "population_size": 50,
        "sigma": 0.1,
        "learning_rate": 0.03,
        "decay": 0.999,
        "num_episode": 1000
    }

    # EAs = EA(config)
    #
    # state = env.reset()
    # print("STATE: ", state.shape)
    # sys.exit(0)
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     print(action)
    #     sys.exit(0)
    #     observation, reward, done, info = env.step(action)
    #     print(observation.shape)
    #     print(reward)
    #     print(done)
    #     print(info)


if __name__ == "__main__":
    t = EA(10000, parallel=10)
    t.main()