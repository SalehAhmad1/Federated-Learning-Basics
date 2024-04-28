import torch
import torch.nn as nn
import torch.optim as optim

from cryptography.hazmat.primitives.asymmetric import rsa

import time

from model import *
from dataset import *
from dataset_utils import *
from model_utils import *

import requests

'''Generating Keys for RSA for the client'''
private_key_rsa = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key_rsa = private_key_rsa.public_key()

'''Serializing Keys for RSA for the client'''
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

# Create a local client model for training and testing on client side
local_client_model = ClientModel(num_features=24, num_labels=5)
local_client_model_layer_keys = [name for name, param in local_client_model.named_parameters()]

# Train the model
local_client_model = train(local_client_model, Train_Dataset, num_epochs=10, lr=0.01)

# Test the model
test(local_client_model, Test_Dataset)

# Setting up NGROK and endpoint to send to the client
NGROK = 'https://fd19-2407-d000-a-223c-5e78-74d2-e03e-f43.ngrok-free.app'

# Get server's public key for RSA
response = requests.get(f'{NGROK}/key')
server_public_key_rsa_bytes = response.json()['key'].encode('utf-8')
server_public_key_rsa = serialization.load_pem_public_key(server_public_key_rsa_bytes, backend=default_backend())

# Encrypt the client model's weights using the functions in model_utils.py
encrypted_base64_weights = encrypt(local_client_model.state_dict(), server_public_key_rsa)

# Send the encrypted weights to the server
'''This will send each client's weights to the server so, they are collected together untill all clients do the'''
url = f'{NGROK}/FED'
data_for_server = {'encrypted_base64_weights': encrypted_base64_weights,'key': public_key_bytes_rsa.decode('utf-8')}
response = requests.post(url, json=data_for_server)

# Check till all clients' weights have been collected
while True:
    url = f'{NGROK}/check_for_wait'
    response = requests.get(url)
    if not response.json()['message'] == 'All weights received':
        time.sleep(3)
    else:
        break

# FedAveraging of model weights
url = f'{NGROK}/aggregate'
data_for_server = {'key': public_key_bytes_rsa.decode('utf-8')}
response = requests.post(url, json=data_for_server)
base64_weights = response.json()['base64_weights'] # Decrypt the model weights and converting to tensors
decrypted_weights = decrypt(base64_weights, private_key_rsa)
aggregated_weights = {k: torch.tensor(v) for k, v in decrypted_weights.items()}

# Load the aggregated weights into the local model
local_client_model.load_state_dict(aggregated_weights)

# Test the newly aggregated model
test(local_client_model, Test_Dataset)