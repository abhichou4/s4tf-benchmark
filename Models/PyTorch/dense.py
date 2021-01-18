import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseModel(nn.Module):

    def __init__(self, inputs, outputs):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(inputs, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, outputs)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
