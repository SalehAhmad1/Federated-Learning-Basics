import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from dataset import *

def load_get_dataset(dataset_name='load_iris'):
    if dataset_name=='load_iris':
        iris = load_iris()
        X, Y = iris.data, iris.target   
        DF = pd.DataFrame(X, columns = iris.feature_names)        
        unique_labels = np.unique(Y)
        str2idx = {col: idx for idx, col in enumerate(unique_labels)}
        idx2str = {idx: col for idx, col in enumerate(unique_labels)}
        DF['target'] = Y
        return DF, str2idx, idx2str 
    
def randomize_dataset(DF):
    random_indices = np.random.permutation(DF.index)
    shuffled_df = DF.iloc[random_indices].reset_index(drop=True)
    return shuffled_df

def split_dataframe_into_train_test(DF):
    train, test = train_test_split(DF, test_size=0.3, random_state=51)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test
    
def split_data(data, n):
    split_size = len(data) // n
    splits = [data[i*split_size:(i+1)*split_size] for i in range(n)]
    return splits

def custom_dataset_from_splits(splits):
    datasets = []
    for split_data in splits:
        dataset = CustomDataset(split_data)
        datasets.append(dataset)
    return datasets(splits)