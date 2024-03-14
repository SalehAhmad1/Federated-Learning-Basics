import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

class ServerModel(BaseModel):
    def __init__(self):
        super(ServerModel, self).__init__()

class ClientModel(BaseModel):
    def __init__(self):
        super(ClientModel, self).__init__()