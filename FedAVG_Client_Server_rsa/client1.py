import torch
import torch.nn as nn
import torch.optim as optim

from cryptography.hazmat.primitives.asymmetric import rsa

import numpy as np
import time

from model import *
from dataset import *
from dataset_utils import *
from model_utils import *

import pickle
import base64

import requests

# Generate RSA key pair
private_key_rsa = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key_rsa = private_key_rsa.public_key()

# Serialize the RSA public key
public_key_bytes_rsa = public_key_rsa.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Load the main dataset, and shuffle it so every split has mix labels
DF, str2idx, idx2str  = load_get_dataset()
DF = randomize_dataset(DF)
TrainDF, TestDF = split_dataframe_into_train_test(DF)
Train_Dataset = CustomDataset(TrainDF, target_col_name='target')
Test_Dataset = CustomDataset(TestDF, target_col_name='target')

local_client_model = ClientModel(num_features=24, num_labels=5)
local_client_model_layer_keys = [name for name, param in local_client_model.named_parameters()]

local_client_model = train(local_client_model, Train_Dataset, num_epochs=10, lr=0.01)
test(local_client_model, Test_Dataset)

#okojk
NGROK = 'https://fd19-2407-d000-a-223c-5e78-74d2-e03e-f43.ngrok-free.app'

# Get the RSA public key from the server
response = requests.get(f'{NGROK}/key')
server_public_key_bytes_rsa = response.json()['key'].encode('utf-8')
server_public_key_rsa = serialization.load_pem_public_key(server_public_key_bytes_rsa, backend=default_backend())

# Encrypt the model weights
base64_weights = encrypt(local_client_model.state_dict(), server_public_key_rsa)

# Federation
url = f'{NGROK}/federation'
data = {
    'base64_weights': base64_weights,
    'key': public_key_bytes_rsa.decode('utf-8'),
}
response = requests.post(url, json=data)
print(response)
print(response.json()['message'])

# Wait for all model weights to be received
while True:
    url = f'{NGROK}/check'
    response = requests.get(url)
    if response.json()['message'] == 'All weights received':
        break
    else:
        time.sleep(5)

# Aggregation
url = f'{NGROK}/aggregate'
data = {
    'base64_weights': base64_weights,
    'key': public_key_bytes_rsa.decode('utf-8'),
}
response = requests.post(url, json=data)

base64_weights = response.json()['base64_weights']

# Decrypt the model weights
decrypted_weights = decrypt(base64_weights, private_key_rsa)

# Convert the aggregated weights back to a dictionary
aggregated_weights = {k: torch.tensor(v) for k, v in decrypted_weights.items()}

# Load the aggregated weights into the model
local_client_model.load_state_dict(aggregated_weights)

# Test the model
test(local_client_model, Test_Dataset)