import torch

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
import glob

from dataset import *

def convert_categorical_to_numerical(df, column_name, str2idx):
    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].map(str2idx)
    return df_copy


def get_dicts(df, column_name):
    unique_values = df[column_name].unique()
    str2idx = {value: idx for idx, value in enumerate(unique_values)}
    idx2str = {idx: value for idx, value in enumerate(unique_values)}
    return str2idx, idx2str

def load_get_dataset():
    dataset_dir = '../Data/'
    csv_files = []
    for file_path in glob.glob(os.path.join(dataset_dir, '*.csv')):
        csv_files.append(os.path.basename(file_path))
    
    num_files = len(csv_files)
    if num_files == 0:
        raise Exception("No CSV file found in the directory.")
    elif num_files > 1:
        raise Exception("More than one CSV file found in the directory.")
    else:
        csv_file = csv_files[0]
        path_csv = os.path.join('..', 'Data', csv_file)
        df = pd.read_csv(path_csv)
        if 'target' in df.columns:
            str2idx,idx2str = get_dicts(df, column_name='target')
            df = convert_categorical_to_numerical(df, column_name='target', str2idx=str2idx)
            return df, str2idx, idx2str
        else:
            raise Exception("No column \"target\" in the csv")
    
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