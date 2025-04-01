# -*- coding: utf-8 -*-
from torch.utils.data import Dataset


class CustomDataset(Dataset): 
    def __init__(self, feature_list, label):
        self.x_data = feature_list
        self.y_data = label

    def __len__(self): 
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y
