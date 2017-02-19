from collections import deque
from PIL import Image
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
        return stack_o, r, done, info

    def reset(self):
        o = self.env.reset()
        for _ in range(self.frame_num):
            self.que.append(o)
        stack_o = np.concatenate([o for o in self.que], axis=0)
        return stack_o

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

class AtariEnv(object):
    def __init__(self, env, input_shape=(84, 84)):
        self.env = env
        self.input_shape = input_shape

    def step(self, action):
        o, r, done, env_info = self.env.step(action)
        o = self._preprocess_obs(o)
        max_o = np.maximum(o, self.last_o)
        self.last_o = o
        r = self._preprocess_r(r)
        return max_o, r, done, env_info
    
    def reset(self):
        o = self.env.reset()
        o = self._preprocess_obs(o)
        self.last_o = o
        return o

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

    def _preprocess_obs(self, obs):
        assert obs.ndim == 3  # (height, width, channel)
        img = Image.fromarray(obs)
        img = img.resize(self.input_shape).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == self.input_shape
        return self._to_float(processed_observation).reshape(1, *self.input_shape)

    def _to_float(self, data):
        """
        int to float
        """
        processed_data = data.astype('float32') / 255.
        return processed_data

    def _preprocess_r(self, reward):
        return np.clip(reward, -1., 1.)
