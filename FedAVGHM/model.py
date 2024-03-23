import torch
import torch.nn as nn
import numpy as np
import phe as paillier

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
        self.initialize_keys()

    def initialize_keys(self):
        keypair = paillier.generate_paillier_keypair(n_length=1024)
        self.pubkey, self.privkey = keypair

    def decrypt(self):
        with torch.no_grad():
            for param in self.parameters():
                # print(param)
                pass

class ClientModel(BaseModel):
    def __init__(self, pub_key, num_features, num_labels):
        self.pubkey = pub_key
        super(ClientModel, self).__init__(num_features, num_labels)

    def encrypt_tensor(self, tensor):
        encrypted = [self.pubkey.encrypt(float(i.item())) for i in tensor.flatten()]
        encrypted_same_shape = np.array(encrypted).reshape(tensor.shape)
        encrypted_same_shape += encrypted_same_shape
        # return torch.tensor(encrypted_same_shape)
    
    def encrypt(self):
        with torch.no_grad():
            for param in self.parameters():
                shape = param.shape
                print(shape)
                print(param.data)
                param.data = self.encrypt_tensor(param.data)
                param.data = param.data.view(shape)
                print(param.data)
                return