from torch.utils.data import Dataset
import numpy as np
import torch

class CMAPSS_Dataset(Dataset):
    def __init__(self, mode='train', data_path=None, subset='13', seq_length=30, max_rul=125, handcrafted=False):

        self.mode = mode
        self.seq_length = seq_length
        self.max_rul = max_rul
        self.handcrafted = handcrafted
        self.unused_cols = [5, 9, 10, 14, 20, 22, 23]

        # Load raw data
        if self.mode == 'train':
            self.data = np.loadtxt(fname=f'{data_path}/train_{subset}.txt', dtype=np.float32)
        elif self.mode == 'test':
            self.data = np.loadtxt(fname=f'{data_path}/test_{subset}.txt', dtype=np.float32)
            self.rul_test = np.loadtxt(fname=f'{data_path}/rul_{subset}.txt', dtype=np.float32)

        # self.sample_num = int(self.data[-1][0])
        self.unique_ids = np.unique(self.data[:, 0])
        self.engine_id_to_index = {engine_id: idx for idx, engine_id in enumerate(self.unique_ids)}
        print(f"Found {len(self.unique_ids)} unique motor IDs in the data set.")

        # Preprocess data
        self.x, self.y = self.__preprocess(self.data)

        if self.handcrafted:
            self.mean_and_coef = self.__handcraft()
            print(self.mean_and_coef.shape)
        else:
            self.mean_and_coef = np.zeros((self.x.shape[0], 2 * self.x.shape[2]))


    def __preprocess(self, data):
        data = np.delete(data, self.unused_cols, axis=1)

        x = []
        y = []


        for id in self.unique_ids:
            ind = np.where(data[:, 0] == id)[0]
            data_temp = data[ind, :]



            if self.mode == 'train':
                for j in range(len(data_temp) - self.seq_length + 1):
                    x.append(data_temp[j: j+self.seq_length, 2:])
                    rul = len(data_temp) - self.seq_length - j
                    if rul > self.max_rul:
                        rul = self.max_rul
                    y.append(rul)

            elif self.mode == 'test':
                if len(data_temp) < self.seq_length:
                    if len(data_temp) == 0:
                        print(f"Skipping engine_no {id} due to empty data_temp.")
                        continue

                    # Interpolation for sequences that are too short
                    data_inter = np.zeros((self.seq_length, data_temp.shape[1]))  
                    for j in range(data_inter.shape[1]):
                        x_old = np.linspace(0, len(data_temp) - 1, len(data_temp), dtype=np.float64)
                        params = np.polyfit(x_old, data_temp[:, j].flatten(), deg=1)
                        k, b = params
                        x_new = np.linspace(0, self.seq_length - 1, self.seq_length, dtype=np.float64)
                        data_inter[:, j] = (x_new * len(data_temp) / self.seq_length * k + b)
                    x.append(data_inter[-self.seq_length:, 2:])              
                else:
                    # Use data without interpolation
                    x.append(data_temp[-self.seq_length:, 2:])
                    
                rul_index = self.engine_id_to_index[id]
                rul = self.rul_test[rul_index]
                if rul > self.max_rul:
                    rul = self.max_rul  # RUL auf max_rul begrenzen
                y.append(rul)
        
        x = np.array(x)
        y = np.array(y)
        # Normalize RUL
        y = y / self.max_rul

        return x, y

    def __handcraft(self):
        mean_and_coef = []
        for i in range(len(self.x)):
            one_sample = self.x[i]
            mean_and_coef.append(self.fea_extract(one_sample))

        mu = np.mean(mean_and_coef, axis=0)
        sigma = np.std(mean_and_coef, axis=0)
        eps = 1e-10
        mean_and_coef = (mean_and_coef - mu) / (sigma + eps)

        return np.array(mean_and_coef)
    
    @staticmethod
    def fea_extract(data):
        fea = []
        x = np.array(range(data.shape[0]))
        for i in range(data.shape[1]):
            fea.append(np.mean(data[:, i]))
            fea.append(np.polyfit(x.flatten(), data[:, i].flatten(), deg=1)[0])
        return fea


    def __len__(self):
        return len(self.x)

    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.mean_and_coef[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )