import argparse
import sys
import os

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

from DeepQNetwork import DeepQNetwork
from statistics import mean

if __name__ == "__main__":


    file = "model"
    nb_ep = 1
    if len(sys.argv) > 1:
        file += str(sys.argv[1])
        if os.path.exists("model/" + file + ".pt"):
            eval_nn = torch.load("model/" + file + ".pt")
            print("z")
    else :
        eval_nn = DeepQNetwork(4, 2)
    if len(sys.argv) > 2:
        nb_ep = int(sys.argv)
    episode_count = 100
    if len(sys.argv) > 2:
        episode_count = int(sys.argv[2])

    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    # args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v1')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'vids'
    env.seed(0)

    rewards = []
    episodes = []
    buffer = []
    buffer_max = 100000
    rewards = []
    for i in range(200):
        s = env.reset()
        ep_reward = 0
        while True:
            output = eval_nn(torch.FloatTensor(s))
            a = int(torch.argmax(output))
            s_, r, done, _ = env.step(a)
            s = s_
            ep_reward += r
            if done:
                break
        rewards.append(ep_reward)
    print(mean(rewards))
    plt.plot(rewards)
    plt.show()
    
    for i in range(nb_ep):
        s = env.reset()
        ep_reward = 0
        while True:
            env.render()
            output = eval_nn(torch.FloatTensor(s))
            a = int(torch.argmax(output))
            s_, r, done, _ = env.step(a)
            s = s_
            ep_reward += r
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), )
                break
    # Close the env and write monitor result info to disk
    env.close()

####################################################################################################################################
