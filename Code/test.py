import sys

# Import the game
import gym
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import time


env = gym.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Create a flag - restart or not

state = env.reset()
print(state.shape)

# Loop through each frame in the game
for step in range(1000):
    # Start the game to begin with
    # Do random actions
    # action = int(input("Enter action:"))
    state, reward, done, info = env.step(2)
    print(reward, done, info)
    sys.exit(0)

    # Show the game on the screen
    env.render()
    time.sleep(0.1)
# Close the game
env.close()