from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class IDSDataset(Dataset):
    def __init__(self, npz_file, name, train_num=None):
        self.data_df = np.load(npz_file)[name]
        self.name = name
        self.train_num = len(self.data_df) if train_num is None else train_num

    def __len__(self):
        return self.train_num

    def __getitem__(self, index):
        temp = self.data_df[index]
        flow_values = torch.FloatTensor(temp[:-1]) * 2 - 1
        label = temp[-1]
        tensor_label = torch.FloatTensor([0, 1]) if label else torch.FloatTensor([1, 0])
        return flow_values, label, tensor_label


def ids_dataloader(npz, name, batch_size, train_num=None, shuffle=True, drop_last=False):
    return DataLoader(IDSDataset(npz_file=npz, name=name, train_num=train_num),
                      batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
