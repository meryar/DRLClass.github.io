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
from wrapper import AtariPreprocessing


#####################  hyper parameters  ####################

EPSILON = .1
MAX_EPISODES = 150
MAX_EP_STEPS = 200
GAMMA = 0.95     # reward discount
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
LEARNING_RATE = .001
REFRESH_GAP = 1000
EXPLORATION_DECAY = 0.995
RENDER = False
ACTION_DIM = 4
ENV_DIM = [4,84,84]

###############################  Agent  ####################################
class Agent():
    def __init__(self):
        self.memory = []
        self.eval_nn = DeepQNetwork(ENV_DIM, ACTION_DIM)
        self.target_nn = DeepQNetwork(ENV_DIM, ACTION_DIM)
        self.optimizer = torch.optim.Adam(self.eval_nn.parameters(),lr=LEARNING_RATE)
        self.criterion = nn.MSELoss(reduction='sum')
        self.counter = 0
        self.eval_nn.conv1 = self.target_nn.conv1
        self.eval_nn.conv2 = self.target_nn.conv2
        self.eval_nn.conv3 = self.target_nn.conv3
        self.eval_nn.f = self.target_nn.f
        self.eval_nn.f2 = self.target_nn.f2

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
            qValues = (self.eval_nn(torch.tensor(s).float().unsqueeze(0)))[a]
            qValues_ = self.target_nn(torch.tensor(s_).float().unsqueeze(0))
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
            self.eval_nn.conv1 = self.target_nn.conv1
            self.eval_nn.conv2 = self.target_nn.conv2
            self.eval_nn.conv3 = self.target_nn.conv3
            self.eval_nn.f = self.target_nn.f
            self.eval_nn.f2 = self.target_nn.f2

    def store_transition(self, value):
        self.memory.append(value)
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.pop(0)

###############################  training  ####################################

if __name__ == '__main__':
    counter = 0
    env = AtariPreprocessing()
    
    file = "model/model"
    if len(sys.argv) > 1:
        file += str(sys.argv[1])
    file += ".pt"

    ag = Agent()

    t1 = time.time()
    for i in range(MAX_EPISODES):
        done = False
        s = env.reset()
        ep_reward = 0
        print("Start Ep ", i)
        while True:
            if random.random() < EPSILON:
                a = random.randint(0,3)
            else:
                output = ag.eval_nn(torch.FloatTensor(s).unsqueeze(0))
                a = int(torch.argmax(output))
            sn, r, done, died = env.step(a)
            # print(sn, r, done)
            ag.store_transition((s, a, sn, r, done))
            s = sn
            
            if len(ag.memory) >= BATCH_SIZE:
                ag.optimize_model(file)
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