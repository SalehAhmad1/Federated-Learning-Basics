import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(nn.Module):
    def __init__(self, shift=0):
        super(BaseModel, self).__init__()
        self.shift=shift
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
    def __init__(self, shift=0):
        super(ServerModel, self).__init__(shift=shift)

    def decrypt(self, shift):
        # Decrypt model parameters (reverse Caesar cipher)
        with torch.no_grad():
            for param in self.parameters():
                param.add_(-shift)

class ClientModel(BaseModel):
    def __init__(self, shift=0):
        super(ClientModel, self).__init__(shift=shift)

    def encrypt(self):
        with torch.no_grad():
            for param in self.parameters():
                param.add_(self.shift)