import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, num_actions, dim_obs, frame_num=4):
        super(Policy, self).__init__()
        self.num_actions = num_actions
        self.dim_obs = dim_obs
        self.frame_num = frame_num
        self.fc1 = nn.Linear(dim_obs*frame_num, 128)
        self.fc2 = nn.Linear(128, 128)
        self.p = nn.Linear(128, num_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.p(x)
        value = self.v(x)
        return F.softmax(policy), value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()


