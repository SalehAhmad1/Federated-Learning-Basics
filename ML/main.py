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
    num_epochs = 10
    lr = 0.01

    # Load the main dataset, and shuffle it so every split has mix labels
    DF, str2idx, idx2str  = load_get_dataset()
    DF = randomize_dataset(DF)
    TrainDF, TestDF = split_dataframe_into_train_test(DF)

    print(TrainDF.target.value_counts())
    print(TestDF.target.value_counts())

    #Number of features, Number of Prediction Labels
    num_features = len(DF.columns) - 1
    num_labels = len(str2idx)

    Preprocessed_Train_DF, Scaler = preprocess_df(DF=TrainDF, scaler_type='MinMaxScaler')
    save_preprocessing_scaler(scaler=Scaler, filepath='./preprocessing_pickles/scaler.pkl')
    Preprocessed_Test_DF, scaler = preprocess_df(DF=TestDF, scaler=Scaler, scaler_type='MinMaxScaler')

    # Create client model
    client_models = ClientModel(num_features=num_features, num_labels=num_labels)

    # Split the data for client
    train_datasets = CustomDataset(TrainDF, target_col_name='target')
    
    # Train the model
    client_model = train(client_models, train_datasets, num_epochs=num_epochs, lr=lr)
    
    #Save the Model  
    save_model(client_model, filepath='./ClientModel.pth')
    
    #Load the Model
    model_loaded = load_model(ClientModel, args=(num_features, num_labels), filepath='./ClientModel.pth')

    #Test the Model
    print('Testing the Server Model')
    test_datasets = [CustomDataset(Preprocessed_Test_DF, target_col_name='target')]
    test(model_loaded, test_datasets)