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
    
    def print_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data)

class ServerModel(BaseModel):
    def __init__(self, num_features,num_labels):
        super(ServerModel, self).__init__(num_features,num_labels)

    def decrypt(self):
        # Decrypt model parameters (ROT13 - inverse shift)
        with torch.no_grad():
            for param in self.parameters():
                param.add_(13)
                param.fmod_(26)

class ClientModel(BaseModel):
    def __init__(self, num_features, num_labels):
        super(ClientModel, self).__init__(num_features, num_labels)

    def encrypt(self):
        # Encrypt model parameters (ROT13 - shift)
        with torch.no_grad():
            for param in self.parameters():
                param.add_(13)
                param.fmod_(26)