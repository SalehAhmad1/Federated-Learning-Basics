import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from model import *
from dataset import *
from dataset_utils import *
from model_utils import *

import pickle
import base64

import requests

# Load the main dataset, and shuffle it so every split has mix labels
DF, str2idx, idx2str  = load_get_dataset()
DF = randomize_dataset(DF)
TrainDF, TestDF = split_dataframe_into_train_test(DF)
Train_Dataset = CustomDataset(TrainDF, target_col_name='target')
Test_Dataset = CustomDataset(TestDF, target_col_name='target')

local_client_model = ClientModel(num_features=24, num_labels=5)
local_client_model_layer_keys = [name for name, param in local_client_model.named_parameters()]

client_model = train(local_client_model, Train_Dataset, num_epochs=10, lr=0.01)
test(client_model, Test_Dataset)

encrypted_weights, key = encrypt_weights_AES(client_model)
ngrok_link = 'https://bd2d-2407-d000-a-223c-5e78-74d2-e03e-f43.ngrok-free.app'
endpoint = '/update_server'
data = {'encrypted_weights': encrypted_weights, 'key': key}
response = requests.post(ngrok_link + endpoint, json=data)
server_combined_bytes_data = response.content
server_weights_encrypted = base64.b64decode(server_combined_bytes_data)
# server_decrypted_weights = pickle.loads(server_weights_encrypted)

original_bytes_data = server_weights_encrypted[:-44]
splitted_bytes_data = original_bytes_data.split(b'--')
original_byte_data_dict = {key: val for key, val in zip(local_client_model_layer_keys, splitted_bytes_data)}
original_key_data = server_weights_encrypted[-44:]

decrypt_server_weights = decrypt_weights_AES(original_byte_data_dict, original_key_data)
local_client_model.load_state_dict(decrypt_server_weights)

print(f'local_client', local_client_model)