import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

def _cnn_to_linear(seq, input_shape=None):
    if isinstance(input_shape, tuple):
        input_shape = list(input_shape)
    if input_shape is None:
        assert False, 'input_shape must be determined'
    for cnn in seq:
        if not isinstance(cnn, nn.Conv2d):
            continue
        kernel_size = cnn.kernel_size
        stride = cnn.stride
        for i, l in enumerate(input_shape):
            input_shape[i] = (l - kernel_size[i] + stride[i])//stride[i]
        channel_size = cnn.out_channels
    return input_shape[0] * input_shape[1] * channel_size

class AtariCNN(nn.Module):
    def __init__(self, frame_num=4, input_shape=(84, 84), out_dim=512):
        super(AtariCNN, self).__init__()
        self.frame_num = frame_num
        self.input_shape = input_shape
        self.out_dim = out_dim
        self.f = nn.Sequential(
                nn.Conv2d(frame_num, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU()
                )
        self.linear_dim = _cnn_to_linear(self.f, input_shape)
        self.fc1 = nn.Linear(self.linear_dim, self.out_dim)

    def forward(self, x):
        x = self.f(x)
        x = x.view(-1, self.linear_dim)
        x = F.relu(self.fc1(x))
        return x

class Policy(nn.Module):
    def __init__(self, num_actions, atari, dim_obs=None, out_dim=512, frame_num=4):
        super(Policy, self).__init__()
        self.num_actions = num_actions
        self.dim_obs = dim_obs
        self.frame_num = frame_num
        self.out_dim = out_dim
        self.atari = atari
        if atari:
            self.head = AtariCNN(frame_num, out_dim=out_dim)
        else:
            self.head = nn.Linear(dim_obs*frame_num, out_dim)
        self.p = nn.Linear(out_dim, num_actions)
        self.v = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = F.relu(self.head(x))
        policy = self.p(x)
        value = self.v(x)
        return F.softmax(policy), value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()


