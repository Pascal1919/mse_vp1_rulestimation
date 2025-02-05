import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CMAPSSDataset(Dataset):
    def __init__(self, mode='train', data_path=None, subset='13', max_rul=125):
        """
        CMAPSS Dataset loader with dynamic KMeans clustering for operating conditions.

        Args:
            mode (str): Decides if train or test data is used.
            data_path: Path to the data folder.
            subset (str): Subset of the dataset to load (e.g., '13').
            max_rul (int): Maximum RUL for the piecewise linear degradation model.
        """
        self.mode = mode
        self.max_rul = max_rul
        self.unused_cols = [2, 3, 4, 5, 9, 10, 14, 20, 22, 23]


        # Load raw data
        if self.mode == 'train':
            self.data = np.loadtxt(fname=f'{data_path}/train_{subset}.txt', dtype=np.float32)
        elif self.mode == 'test':
            self.data = np.loadtxt(fname=f'{data_path}/test_{subset}.txt', dtype=np.float32)
            self.rul_test = np.loadtxt(fname=f'{data_path}/rul_{subset}.txt', dtype=np.float32)

        self.unique_ids = np.unique(self.data[:, 0])
        self.engine_id_to_index = {engine_id: idx for idx, engine_id in enumerate(self.unique_ids)}
        print(f"Found {len(self.unique_ids)} unique motor IDs in the data set.")

        # Preprocess data
        self.x, self.y, self.t = self._preprocess(self.data)


    def _preprocess(self, data):

        data = np.delete(data, self.unused_cols, axis=1)

        if self.mode == 'train':
            x = data
        elif self.mode == 'test':
            x = []

        y = []
        t = []

        for id in self.unique_ids:
            ind = np.where(data[:, 0] == id)[0]
            data_temp = data[ind, :]
            cycles = data_temp[:, 1]
            
            # Compute RUL for training data
            if self.mode == 'train':
                last_cycle = np.max(data_temp[:, 1])
                rul = np.array([last_cycle - cycle for cycle in cycles])
                rul = np.minimum(rul, self.max_rul)
                t.append(cycles)

            # Compute RUL for test data
            elif self.mode == 'test':
                x_last = data_temp[-1,:]
                t_last = cycles[-1]
                rul_index = self.engine_id_to_index[id]
                rul = self.rul_test[rul_index]
                rul = np.minimum(rul, self.max_rul)
                x.append(x_last)
                t.append(t_last)

            y.append(rul)
            

        x = np.array(x)
        print(x.shape)
        x = x[:, 2:]
        if self.mode == 'train':
            y = np.concatenate(y, axis=0)
            t = np.concatenate(t, axis=0)
        elif self.mode == 'test':
            y = np.array(y)
            t = np.array(t)
        # Normalize time (cycles)
        t = t / t.max()

        return x, y, t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
            return (
                torch.tensor(self.x[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
                torch.tensor(self.t[idx], dtype=torch.float32),
            )

