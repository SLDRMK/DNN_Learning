import os
import json
import torch
import torch.nn as nn

from .io import load_data

class DefaultDataset(nn.Module):
    def __init__(
        self,
        root='./datasets/default',
        transform=None,
        split='train',
        split_file="split.json"):
        super(DefaultDataset, self).__init__()

        self.root = root
        self.transform = transform
        self.split = split
        self.split_file = os.path.join(root, split_file) if split_file else None

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

        self.data = self.data[split]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        single_data = self.data[idx]
        data = {}

        for key in single_data:
            data_path = os.path.join(self.root, single_data[key])
            data[key] = load_data(data_path)
        
        data = self.transform(data)
        return data