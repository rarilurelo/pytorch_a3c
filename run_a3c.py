import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp

from async_rmsprop import AsyncRMSprop
from policy import Policy
from wrapper_env import WrapperEnv

FRAME_HEIGHT = 84
FRAME_WIDTH = 84

def preprocess(pre_o, o):
    pro_o = np.maximum(pre_o, o)
    pro_o = resize(rgb2gray(pro_o), (FRAME_WIDTH, FRAME_HEIGHT))
    pro_o = np.reshape(pro_o, (1, FRAME_WIDTH, FRAME_HEIGHT))
    return pro_o

env = gym.make('Breakout-v0')
env = WrapperEnv(env, 4, preprocess)

global_policy = Policy(env.action_space.n)
global_policy.share_memory()
local_policy = Policy(env.action_space.n)

lr = 0.00025
optimizer = AsyncRMSprop(global_policy.parameters(), local_policy.parameters(), lr=lr)

global_t = torch.IntTensor(1).share_memory_()

def train(rank, nb_epoch, local_t_max, global_t):
    o = env.reset()
    step = 0
    sum_rewards = 0
    while global_t[0] < nb_epoch:
        local_policy.sync(global_policy)
        observations = []
        actions = []
        values = []
        rewards = []
        probs = []
        R = 0
        for i in range(local_t_max):
            global_t += 1
            step += 1
            p, v = local_policy(o)
            a = p.multinomial()
            o, r, done, _ = env.step(a.data.squeeze()[0])
            if rank == 0:
                sum_rewards += r
                env.render()
            observations.append(o)
            actions.append(a)
            values.append(v)
            rewards.append(r)
            probs.append(p)
            if done:
                o = env.reset()
                if rank == 0:
                    print('total reward', sum_rewards)
                    sum_rewards = 0
                    print('probs', probs)
                    print('global_t', global_t[0])
                break
        else:
            _, v = local_policy(o)
            R += v.data.squeeze()[0]

        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        if len(returns) > 1:
            returns = (returns-returns.mean()) / (returns.std()+1e-4)
        v_loss = 0
        entropy = 0
        #policy_loss = 0
        for a, v, p, r in zip(actions, values, probs, returns):
            a.reinforce(r - v.data.squeeze())
            #_policy_loss = (p + 1e-4).log().squeeze()[a.data.squeeze()[0]] #* (r -v.data)
            #_policy_loss *= r - v.data.squeeze()[0]
            #policy_loss += _policy_loss
            _v_loss = nn.MSELoss()(v, Variable(torch.Tensor([r])))
            v_loss += _v_loss
            entropy += (p * (p + 1e-4).log()).sum()
        v_loss = v_loss * 0.5 * 0.5
        entropy = entropy * 0.1
        optimizer.zero_grad()
        #total_loss = v_loss + entropy + policy_loss
        #print(total_loss)
        #total_loss.backward()
        final_node = [v_loss, entropy] + actions
        gradients = [torch.ones(1), torch.ones(1)] + [None] * len(actions)
        autograd.backward(final_node, gradients)
        new_lr = (nb_epoch - global_t[0]) / nb_epoch * lr
        optimizer.step(new_lr)

processes = []
for rank in range(4):
    p = mp.Process(target=train, args=(rank, 10000000, 10, global_t))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
