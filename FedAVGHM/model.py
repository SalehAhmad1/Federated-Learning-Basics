import torch
import torch.nn as nn
import numpy as np
import phe as paillier
import os
import pickle

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

    def decrypt(self, model_name='None'):
        with torch.no_grad():
            npy = np.load(f"./encrypted_models/{model_name}.npy")
            for idx,param in enumerate(self.parameters()):
                shape_of_param = param.shape
                decrypted = [self.privkey.decrypt(i) for i in npy[idx].flatten()]
                decrypted = np.array(decrypted).reshape(shape_of_param)
                param.data = torch.from_numpy(decrypted)

class ClientModel(BaseModel):
    def __init__(self, pub_key, num_features, num_labels):
        self.pubkey = pub_key
        super(ClientModel, self).__init__(num_features, num_labels)

    def encrypt_tensor(self, tensor):
        encrypted = [self.pubkey.encrypt(float(i.item())) for i in tensor.flatten()]
        return encrypted
    
    def encrypt(self, model_name='None'):
        if not os.path.exists('./encrypted_models'):
            os.makedirs('./encrypted_models')
        with torch.no_grad():
            All_Encrypted = []
            for idx,param in enumerate(self.parameters()):
                shape = param.shape
                encryption = self.encrypt_tensor(param.data)
                # encryption = np.array(encryption).reshape(shape)
                # print(f'Shape of encrypted tensor: {np.shape(encryption)}')
                All_Encrypted.append(np.array(encryption))
            np.save(f"./encrypted_models/{model_name}.npy", np.array(All_Encrypted))