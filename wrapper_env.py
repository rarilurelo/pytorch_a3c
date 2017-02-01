from collections import deque
import numpy as np
from torch.autograd import Variable
import torch

class StackEnv(object):
    def __init__(self, env, frame_num):
        self.env = env
        self.frame_num = frame_num
        self.que = deque(maxlen=self.frame_num)

    def step(self, action):
        o, r, done, info = self.env.step(action)
        self.que.append(o)
        stack_o = np.concatenate([o for o in self.que], axis=0)
        return Variable(torch.from_numpy(stack_o).float().unsqueeze(0)), r, done, info

    def reset(self):
        o = self.env.reset()
        for _ in range(self.frame_num):
            self.que.append(o)
        stack_o = np.concatenate([o for o in self.que], axis=0)
        return Variable(torch.from_numpy(stack_o).float().unsqueeze(0))

    def render(self):
        self.env.render()

    def seed(self, s):
        self.env.seed(s)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space



