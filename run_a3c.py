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
from wrapper_env import StackEnv, AtariEnv
import logger


def train(rank, global_policy, local_policy, optimizer, env, global_t, args):
    o = env.reset()
    step = 0
    sum_rewards = 0
    max_sum_rewards = 0
    vs = []
    entropies = []
    sum_rewards = 0
    while global_t[0] < args.epoch:
        local_policy.sync(global_policy)
        observations = []
        actions = []
        values = []
        rewards = []
        probs = []
        R = 0
        for i in range(args.local_t_max):
            global_t += 1
            step += 1
            p, v = local_policy(Variable(torch.from_numpy(o).float()).unsqueeze(0))
            a = p.multinomial()
            o, r, done, _ = env.step(a.data.squeeze()[0])
            if rank == 0:
                sum_rewards += r
                if args.render:
                    env.render()
            observations.append(o)
            actions.append(a)
            values.append(v)
            rewards.append(r)
            probs.append(p)
            if done:
                o = env.reset()
                if rank == 0:
                    print('----------------------------------')
                    print('total reward of the episode:', sum_rewards)
                    print('----------------------------------')
                    if args.save_mode == 'all':
                        torch.save(local_policy, os.path.join(args.log_dir, args.save_name+"_{}.pkl".format(global_t[0])))
                    elif args.save_mode == 'last':
                        torch.save(local_policy, os.path.join(args.log_dir, args.save_name+'.pkl'))
                    elif args.save_mode == 'max':
                        if max_sum_rewards < sum_rewards:
                            torch.save(local_policy, os.path.join(args.log_dir, args.save_name+'.pkl'))
                            max_sum_rewards = sum_rewards
                    step = 0
                break
        else:
            _, v = local_policy(Variable(torch.from_numpy(o).unsqueeze(0).float()))
            R += v.data.squeeze()[0]

        returns = []
        for r in rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        #if len(returns) > 1:
        #    returns = (returns-returns.mean()) / (returns.std()+args.eps)
        v_loss = 0
        entropy = 0
        for a, v, p, r in zip(actions, values, probs, returns):
            a.reinforce(r - v.data.squeeze())
            _v_loss = nn.MSELoss()(v, Variable(torch.Tensor([r])))
            v_loss += _v_loss
            entropy += -(p * (p + args.eps).log()).sum()
        v_loss = v_loss * 0.5 * args.v_loss_coeff
        entropy = entropy * args.entropy_beta
        loss = v_loss - entropy
        vs.append(v_loss.data.numpy())
        entropies.append(entropy.data.numpy())
        if rank == 0 and done:
            logger.record_tabular_misc_stat('Entropy', entropies)
            logger.record_tabular_misc_stat('V', vs)
            logger.record_tabular('reward', sum_rewards)
            logger.record_tabular('step', global_t[0])
            logger.dump_tabular()
            del vs[:]
            del entropies[:]
            sum_rewards = 0
            print(probs[0])
        optimizer.zero_grad()
        final_node = [loss] + actions
        gradients = [torch.ones(1)] + [None] * len(actions)
        autograd.backward(final_node, gradients)
        new_lr = (args.epoch - global_t[0]) / args.epoch * args.lr
        optimizer.step(new_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch a3c')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--monitor', action='store_true',
                        help='save the rendered video')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='save dir')
    parser.add_argument('--epoch', type=int, default=10000000, metavar='N',
                        help='training epoch number')
    parser.add_argument('--local_t_max', type=int, default=5, metavar='N',
                        help='bias variance control parameter')
    parser.add_argument('--entropy_beta', type=float, default=0.01, metavar='E',
                        help='coefficient of entropy')
    parser.add_argument('--v_loss_coeff', type=float, default=0.5, metavar='V',
                        help='coefficient of value loss')
    parser.add_argument('--frame_num', type=int, default=4, metavar='N',
                        help='number of frames you use as observation')
    parser.add_argument('--lr', type=float, default=7e-4, metavar='L',
                        help='learning rate')
    parser.add_argument('--env', type=str, default='Breakout-v0',
                        help='Environment')
    parser.add_argument('--atari', action='store_true',
                        help='atari environment')
    parser.add_argument('--num_process', type=int, default=8, metavar='n',
                        help='number of processes')
    parser.add_argument('--eps', type=float, default=0.01, metavar='E',
                        help='epsilon minimum log or std')
    parser.add_argument('--save_name', type=str, default='exp', metavar='N',
                        help='define save name')
    parser.add_argument('--save_mode', type=str, default='max', metavar='S',
                        help='save mode. all or last or max')
    args = parser.parse_args()
    logger.add_tabular_output(os.path.join(args.log_dir, 'progress.csv'))
    assert not (args.env == 'Breakout-v0' and not args.atari), 'You should use --atari option'
    logger.log_parameters_lite(os.path.join(args.log_dir, 'params.json'), args)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    env = gym.make(args.env)
    if args.monitor:
        env = wrappers.Monitor(env, args.log_dir, force=True)
    if args.atari:
        env = AtariEnv(env)
    env = StackEnv(env, args.frame_num)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    global_policy = Policy(env.action_space.n, atari=args.atari,
            dim_obs=env.observation_space.shape[0], out_dim=512, frame_num=args.frame_num)
    global_policy.share_memory()
    local_policy = Policy(env.action_space.n, atari=args.atari,
            dim_obs=env.observation_space.shape[0], out_dim=512, frame_num=args.frame_num)

    optimizer = AsyncRMSprop(global_policy.parameters(), local_policy.parameters(), lr=args.lr, eps=args.eps)

    global_t = torch.LongTensor(1).share_memory_()
    global_t.zero_()
    processes = []
    for rank in range(args.num_process):
        p = mp.Process(target=train, args=(rank, global_policy, local_policy, optimizer, env, global_t, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
