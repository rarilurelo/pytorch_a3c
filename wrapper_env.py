from collections import deque
import numpy as np
from torch.autograd import Variable
import torch

class WrapperEnv(object):
    def __init__(self, env, frame_num, preprocess=None):
        self.env = env
        self.frame_num = frame_num
        self.preprocess = preprocess
        self.que = deque(maxlen=self.frame_num)

    def step(self, action):
        o, r, done, info = self.env.step(action)
        if not self.preprocess is None:
            pro_o = self.preprocess(self.last_o, o)
        self.last_o = o
        self.que.append(pro_o)
        o_numpy = np.concatenate([o for o in self.que], axis=0)
        return Variable(torch.from_numpy(o_numpy).float().unsqueeze(0)), r, done, info

    def reset(self):
        o = self.env.reset()
        self.last_o = o
        if not self.preprocess is None:
            pro_o = self.preprocess(o, o)
        for _ in range(self.frame_num):
            self.que.append(pro_o)
        o_numpy = np.concatenate([o for o in self.que], axis=0)
        return Variable(torch.from_numpy(o_numpy).float().unsqueeze(0))

    def render(self):
        self.env.render()

    @property
    def action_space(self):
        return self.env.action_space



