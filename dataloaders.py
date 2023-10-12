# Data loader for NSD data
import os
import numpy as np
import torch
import torch.utils.data as data


class NSDDataLoader(data.Dataset):
    """NSD data loader."""

    def __init__(self, data_path, set, transform=None):
        """
        Args:
            data_path (string): Path to the data folder.
            set (string): Set to load. Options: train, test
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path
        self.set = set
        self.transform = transform

        # Load data
        self.data = np.load(os.path.join(data_path, f'nsd_{set}_data.npy'))
        self.labels = np.load(os.path.join(data_path, f'nsd_{set}_labels.npy'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'data': self.data[idx], 'labels': self.labels[idx]}

        if self.transform:
            sample = self.transform(sample)
            

        return sample