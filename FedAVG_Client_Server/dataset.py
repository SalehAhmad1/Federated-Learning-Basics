import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, DF, target_col_name=None):
        self.DF = DF.reset_index(drop=True)
        self.Y = self.DF[target_col_name]
        self.X = self.DF.drop(target_col_name, axis=1)

    def __len__(self):
        return len(self.DF)

    def __getitem__(self, idx):
        return self.X.iloc[idx, :], self.Y[idx]