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

if __name__ == "__main__":
    # Define parameters
    num_clients = 5
    num_epochs = 10
    lr = 0.01

    # Load the main dataset, and shuffle it so every split has mix labels
    DF, str2idx, idx2str  = load_get_dataset()
    DF = randomize_dataset(DF)
    TrainDF, TestDF = split_dataframe_into_train_test(DF)

    #Number of features, Number of Prediction Labels
    num_features = len(DF.columns) - 1
    num_labels = len(str2idx)

    # Create server model
    server_model = ServerModel(num_features=num_features, num_labels=num_labels)

    # Create client models
    client_models = [ClientModel(num_features=num_features, num_labels=num_labels) for _ in range(num_clients)]

    # Split the data for clients
    train_data_splits = split_data(TrainDF, num_clients)
    tuples = [preprocess_df(DF=split, scaler_type='MinMaxScaler') for split in train_data_splits]
    preprocessed_train_data_splits, Scalers = zip(*tuples)
    save_preprocessing_scaler(scaler=Scalers[0], filepath='./preprocessing_pickles/scaler.pkl')
    train_datasets = [CustomDataset(split, target_col_name='target') for split in preprocessed_train_data_splits]
    preprocessed_test_data, scaler = preprocess_df(DF=TestDF, scaler=Scalers[0], scaler_type='MinMaxScaler')

    # Train the models
    server_model = train(server_model, client_models, train_datasets, num_epochs=num_epochs, lr=lr)
    
    #Save the Server Model  
    save_model(server_model, filepath='./models/ServerModel.pth')
    
    #Load the Server Model
    server_model_loaded = load_model(ServerModel, args=(num_features, num_labels), filepath='./models/ServerModel.pth')

    #Test the Server Model
    test_dataset = CustomDataset(preprocessed_test_data, target_col_name='target')
    test(server_model_loaded, test_dataset)