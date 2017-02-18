import os
import numpy as np
import argparse
import gym
from gym import wrappers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.multiprocessing as mp

from async_rmsprop import AsyncRMSprop
from policy import Policy
from wrapper_env import StackEnv


def test(policy, args):
    step = 0
    sum_rewards = 0
    while step < args.num_rollout:
        done = False
        o = env.reset()
        while not done:
            p, v = policy(o)
            a = p.multinomial()
            o, r, done, _ = env.step(a.data.squeeze()[0])
            sum_rewards += r
            if args.render:
                env.render()
                import time
                time.sleep(1)
        print('----------------------------------')
        print('total reward of the episode:', sum_rewards)
        print('----------------------------------')
        sum_rewards = 0
        step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch a3c load model and test')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--monitor', action='store_true',
                        help='save the rendered video')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='save dir')
    parser.add_argument('--frame_num', type=int, default=4, metavar='N',
                        help='number of frames you use as observation')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Environment')
    parser.add_argument('--model_path', type=str, default='./log_dir/exp.pkl', metavar='S',
                        help='path of saved model')
    parser.add_argument('--num_rollout', type=int, default=6, metavar='N',
                        help='number of rollout')
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    env = gym.make(args.env)
    if args.monitor:
        env = wrappers.Monitor(env, args.log_dir, force=True)
    env = StackEnv(env, args.frame_num)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = torch.load(args.model_path)

    test(policy, args)

