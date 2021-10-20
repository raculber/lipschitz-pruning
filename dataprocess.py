import numpy as np
import pandas as pd


from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class points_dataset(Dataset):
    def __init__(self , mode = 'train'):
        train = pd.read_csv("{}_points.csv".format(mode), header = None)
        train_labels = train[1].values
        train = train[0].values
        self.datalist = torch.from_numpy(train)
        self.labels = torch.from_numpy(train_labels)
    def __getitem__(self, index):
        current_point = self.datalist[index]
        current_label = self.labels[index]
        return  current_point, current_label
    def __len__(self):
        return self.datalist.shape[0]  

def load_data():
    train_data = points_dataset()
    train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = 80, shuffle = True)
    test_data = points_dataset(mode = 'test')
    test_data = torch.utils.data.DataLoader(dataset = test_data , batch_size = 10000 , shuffle = True)
    return train_data, test_data

