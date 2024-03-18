import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self, num_features, num_labels):
        super(BaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_labels),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

class ServerModel(BaseModel):
    def __init__(self, num_features, num_labels):
        super(ServerModel, self).__init__(num_features, num_labels)

class ClientModel(BaseModel):
    def __init__(self, num_features, num_labels):
        super(ClientModel, self).__init__(num_features, num_labels)