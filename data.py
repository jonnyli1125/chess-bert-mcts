import os
import glob

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pytorch_lightning as pl
import h5py
import numpy as np
from numpy.random import randint, random


class BertMLMData(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        paths = glob.glob(os.path.join(self.dataset_dir, '*.h5'))
        train_paths = paths[:-1]
        val_path = paths[-1]
        train_datasets = [BertMLMDataset(train_path)
                          for train_path in train_paths]
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = BertMLMDataset(val_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class BertMLMDataset(Dataset):
    def __init__(self, data_path):
        self.data = h5py.File(data_path, 'r')['policy_value_labels']
        self.mask_token_id = 17

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        inputs = np.array([x[f'state_{i}'] for i in range(len(x) - 3)],
                          dtype=np.int16)
        labels = inputs.copy()

        # select 15% of tokens to mask, apply padding to unmasked tokens
        masked_indices = random(labels.shape) < 0.15
        labels[~masked_indices] = -100

        # mask 80% of selected tokens
        indices_replaced = (random(labels.shape) < 0.8) & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # replace 10% with random tokens
        indices_random = (random(labels.shape) < 0.5) & masked_indices
        indices_random = indices_random & ~indices_replaced
        random_words = randint(0, self.mask_token_id, labels.shape)
        inputs[indices_random] = random_words[indices_random]

        # leave remaining 10% untouched, return item
        item = {'input_ids': torch.tensor(inputs, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long)}
        return item


class BertPolicyValueData(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        paths = glob.glob(os.path.join(self.dataset_dir, '*.h5'))
        train_paths = paths[:-1]
        val_path = paths[-1]
        train_datasets = [BertPolicyValueDataset(train_path)
                          for train_path in train_paths]
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = BertPolicyValueDataset(val_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class BertPolicyValueDataset(Dataset):
    def __init__(self, data_path):
        self.data = h5py.File(data_path, 'r')['policy_value_labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        state = [x[f'state_{i}'] for i in range(len(x) - 3)]
        item = {'input_ids': torch.tensor(state, dtype=torch.long),
                'move': torch.tensor(x['move'], dtype=torch.long),
                'value': torch.tensor(x['value'], dtype=torch.float),
                'result': torch.tensor(x['result'], dtype=torch.long)}
        return item
