# TODO create a class for the PPO algorithm

import neat-python as neat

class PPO():
    def __init__(self):
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    'config-feedforward')
        self.p = neat.Population(self.config)
        self.p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.p.add_reporter(stats)
        self.p.add_reporter(neat.Checkpointer(5))
        self.winner = self.p.run(self.eval_genomes, 300)
        print('\nBest genome:\n{!s}'.format(winner))
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(xor_inputs, xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0] - xo[0]) ** 2
                genome.fitness -= (output[1] - xo[1]) ** 2

if __name__ == "__main__":
    PPO()