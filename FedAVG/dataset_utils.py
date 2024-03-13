import torch

from sklearn.datasets import load_iris

import pandas as pd

from dataset import *

def load_get_dataset(dataset_name='load_iris'):
    if dataset_name=='load_iris':
        iris = load_iris()
        X, Y = iris.data, iris.target   
        DF = pd.DataFrame(X, columns = iris.feature_names)        
        str2idx = {col: idx for idx, col in enumerate(iris.feature_names)}
        idx2str = {idx: col for idx, col in enumerate(iris.feature_names)}
        DF['target'] = Y
        return DF, str2idx, idx2str 
    
    
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
    

