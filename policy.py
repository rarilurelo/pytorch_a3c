import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

class Policy(nn.Module):
    def __init__(self, num_actions):
        super(Policy, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(4, 32, 8, (4, 4))
        self.conv2 = nn.Conv2d(32, 64, 4, (2, 2))
        self.conv3 = nn.Conv2d(64, 64, 3, (1, 1))
        self.fc1 = nn.Linear(64*7*7, 512)
        self.p = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        policy = self.p(x)
        value = self.v(x)
        return F.softmax(policy), value

    def sync(self, global_module):
        for p, gp in zip(self.parameters(), global_module.parameters()):
            p.data = gp.data.clone()


