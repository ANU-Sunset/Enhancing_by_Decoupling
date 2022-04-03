import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
import os

"""
The input of this module is a 256 vector which is the feature representation of AE module
"""


class NetData(Dataset):
    def __init__(self):
        file_list = '../engn8536/Datasets/feature_map/DA_LOSS_training'
        files= os.listdir(file_list)
        self.items = None
        for pt_file in files:
            if self.items == None:
                self.items = torch.load(file_list+'/'+pt_file)
            else:
                self.items = torch.cat((torch.load(file_list+'/'+pt_file),self.items),0)
        self.mu = sum(self.items) / self.items.size(0)
        self.var = torch.sqrt(torch.var(self.items, dim=0))
        self.items = (self.items - self.mu) / self.var

    def __getitem__(self, idx):
        # x = torch.load(self.pt_path,map_location=torch.device('cpu'))
        # return x
        return self.items[idx, :]

    def __len__(self):
        return self.items.size(0)

    def get_all_items(self):
        return self.items
    
    def find_original(self, x):
        y = x * self.var + self.mu
        return y


