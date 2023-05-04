class RANDOM():
    def __init__(self, config, env, render, printing):
        self.config = config
        self.env = env
        self.render = render
        self.printing = printing

    def get_action(self, observation=None, theta=None):
        try:
            action = self.env.action_space.sample()
            return action
        except:
            raise ValueError

    def get_reward(self, env, theta=None, render=False):
        raise NotImplementedError

    def train(self):
        self.env.reset()
        done = False
        
        while not done:
            action = self.get_action()
            observation, reward, done, info = self.env.step(action)
            if self.render: self.env.render()
            if self.printing:
                print(observation.shape)
                print(reward)
                print(done)
                print(info)
