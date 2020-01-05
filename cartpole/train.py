'''
torch = 0.41
'''
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import random
from gym import wrappers, logger
from DeepQNetwork import DeepQNetwork
import matplotlib.pyplot as plt
from random import choices


#####################  hyper parameters  ####################

EPSILON = 0.1
MAX_EPISODES = 150
MAX_EP_STEPS = 200
GAMMA = 0.95     # reward discount
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
LEARNING_RATE = .001
REFRESH_GAP = 1000
RENDER = False
ACTION_DIM = 2
ENV_DIM = 4
ENV_NAME = 'CartPole-v1'

###############################  Agent  ####################################
class Agent():
    def __init__(self):
        self.memory = []
        self.eval_nn = DeepQNetwork(ENV_DIM, ACTION_DIM)
        self.target_nn = DeepQNetwork(ENV_DIM, ACTION_DIM)
        self.optimizer = torch.optim.Adam(self.eval_nn.parameters(),lr=LEARNING_RATE)
        self.criterion = nn.MSELoss(reduction='sum')
        self.counter = 0
        self.target_nn.fc1 = self.eval_nn.fc1
        self.target_nn.fc2 = self.eval_nn.fc2
        self.target_nn.out = self.eval_nn.out

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.eval_nn(s)[0].detach() # ae（s）

    def getMemory(self):
        return self.memory
        
    def optimize_model(self, file):
        batch = random.sample(self.memory, BATCH_SIZE)

        # batch_s = torch.FloatTensor([batch[0][0]])
        # batch_a = torch.LongTensor([batch[0][1]])
        # batch_r = torch.FloatTensor([batch[0][3]])
        # batch_s_ = torch.FloatTensor([batch[0][2]])
        # batch_d = torch.FloatTensor([batch[0][4]])
        # for i in range(1,len(batch)):
        #     batch_s = torch.cat((batch_s, torch.FloatTensor([batch[i][0]])))
        #     batch_a = torch.cat((batch_a, torch.LongTensor([batch[i][1]])))
        #     batch_r = torch.cat((batch_r, torch.FloatTensor([batch[i][3]])))
        #     batch_s_ = torch.cat((batch_s_, torch.FloatTensor([batch[i][2]])))
        #     batch_d = torch.cat((batch_d, torch.FloatTensor([batch[i][4]])))
        # qValues = torch.max(self.eval_nn(batch_s).gather(1, batch_a), dim=1)[0]
        # qValues_ = self.target_nn(batch_s_)
        # qValues_target = torch.max(qValues_, dim = 1)[0]
        # qValues_target *= GAMMA
        # qValues_target = qValues_target * (1 - batch_d)
        # JO = torch.pow(qValues - (batch_r + qValues_target),2)
        for s, a, s_, r, done in batch:
            qValues = (self.eval_nn(torch.tensor(s).float()))[a]
            qValues_ = self.target_nn(torch.tensor(s_).float())
            qValues_target = GAMMA * torch.max(qValues_)
            JO = pow(qValues - (r + (qValues_target * (1 -done))), 2)
            loss = self.criterion(qValues, JO)
            self.optimizer.zero_grad()
            if i != BATCH_SIZE - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.optimizer.step()
        self.counter += 1
        if self.counter % REFRESH_GAP == 0:  
            torch.save(self.eval_nn, file)
            self.target_nn.fc1 = self.eval_nn.fc1
            self.target_nn.fc2 = self.eval_nn.fc2
            self.target_nn.out = self.eval_nn.out

    def store_transition(self, value):
        self.memory.append(value)
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.pop(0)

###############################  training  ####################################

if __name__ == '__main__':
    counter = 0
    env = gym.make(ENV_NAME)
    env.seed(0)

    file = "model/model"
    if len(sys.argv) > 1:
        file += str(sys.argv[1])
    file += ".pt"
    print(1 * True)
    print(1 * False)

    ag = Agent()

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        while True:
            if random.random() < EPSILON:
                a = random.randint(0,1)
            else:
                output = ag.eval_nn(torch.FloatTensor(s))
                a = int(torch.argmax(output))

            sn, r, done, _ = env.step(a)
            if done :
                r = -10
            ag.store_transition((s, a, sn, r, done))
            if len(ag.memory) >= BATCH_SIZE:
                ag.optimize_model(file)
            s = sn
            ep_reward += r
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), )
                break
    print('Running time: ', time.time() - t1)
    rewards = []
    for i in range(200):
        s = env.reset()
        ep_reward = 0
        while True:

            output = ag.eval_nn(torch.FloatTensor(s))
            a = int(torch.argmax(output))
            s_, r, done, _ = env.step(a)
            s = s_
            ep_reward += r
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), )
                break
        rewards.append(ep_reward)
    plt.plot(rewards)
    plt.show()