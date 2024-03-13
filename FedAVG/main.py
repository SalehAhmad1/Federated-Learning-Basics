import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from model import *
from dataset import *
from dataset_utils import *
from model_utils import *

import warnings
warnings.filterwarnings('ignore')


# Example usage
if __name__ == "__main__":
    # Define parameters
    num_clients = 5
    num_epochs = 10
    lr = 0.01

    # Create server model
    server_model = ServerModel()

    # Create client models
    client_models = [ClientModel() for _ in range(num_clients)]

    # Load the main dataset, and shuffle it so every split has mix labels
    DF, str2idx, idx2str  = load_get_dataset(dataset_name='load_iris')
    random_indices = np.random.permutation(DF.index)
    shuffled_df = DF.iloc[random_indices].reset_index(drop=True)
    DF = shuffled_df
    
    # Split the data for clients
    data_splits = split_data(DF, num_clients)
    
    datasets = [CustomDataset(split, target_col_name='target') for split in data_splits]
    
    # Train the models
    train(server_model, client_models, datasets, num_epochs=num_epochs, lr=lr)
    
    #Save the Server Model
    save_model(server_model, filepath='./ServerModel.pth')
    
    #Load the Server Model
    server_model_loaded = load_model(ServerModel, filepath='./ServerModel.pth')
    
    #INfer on random data
    Random_Data = torch.randn(5, 4)
    Results = infer(server_model, Random_Data, idx2str)
    for idx,result in enumerate(Results):
        print(f'The predicted label for the test datapoint:{idx} is:{result}')