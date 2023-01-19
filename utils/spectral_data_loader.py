import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader


LABELS = {
    0: "asphalt", 
    1: "brick", 
    2: "carpet", 
    3: "concrete", 
    4: "grass", 
    5: "gravel", 
    6: "ice", 
    7: "mulch", 
    8: "sand", 
    9: "tile", 
    10: "turf"
}

LABELS_REVERSED = {
    "asphalt": 0, 
    "brick": 1, 
    "carpet": 2, 
    "concrete": 3, 
    "grass": 4, 
    "gravel": 5, 
    "ice": 6, 
    "mulch": 7, 
    "sand": 8, 
    "tile": 9, 
    "turf": 10
}


class SpectralData:

    def __init__(self, data_path, train_test_val_split, batch_size,
                 num_workers, load_data=False):

        self.n_classes = len(LABELS)
        self.data_path = data_path

        if load_data:
            use_path = os.path.abspath(os.path.join(self.data_path,'..','data_loaders'))
            self.train_loader = torch.load(f'{use_path}/spectral_train.dl')
            self.val_loader = torch.load(f'{use_path}/spectral_val.dl')
            self.test_loader = torch.load(f'{use_path}/spectral_test.dl')
            self.N_train = len(self.train_loader.dataset)
            self.N_val = len(self.val_loader.dataset)
            self.N_test = len(self.test_loader.dataset)
            self.N_total = self.N_train + self.N_val + self.N_test
            self.data_dim = self.train_loader.dataset[0][0].shape[0]

        else:
            self.train_split, self.val_split, self.test_split = train_test_val_split

            # don't keep first and last few pieces of spectral data vector
            self.min_ind = 250
            self.max_ind = 1800

            data, labels = self.get_data()

            self.N_total, self.data_dim = data.shape
            self.N_train = int(self.train_split * self.N_total)
            self.N_val = int(self.val_split * self.N_total)
            self.N_test = self.N_total - self.N_train - self.N_val

            data = [[data[i, :], labels[i]] for i in range(self.N_total)]
            train_data, val_data, test_data = random_split(
                data, [self.N_train, self.N_val, self.N_test], 
                generator=torch.Generator().manual_seed(4)
            )

            self.train_loader = DataLoader(train_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
            self.val_loader = DataLoader(val_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)

            use_path = os.path.abspath(os.path.join(self.data_path,'..','data_loaders'))
            torch.save(self.train_loader, f'{use_path}/spectral_train.dl')
            torch.save(self.val_loader, f'{use_path}/spectral_val.dl')
            torch.save(self.test_loader, f'{use_path}/spectral_test.dl')

        print(f"Total number of samples: {self.N_total}")
        print(f"Number of training samples: {self.N_train}")
        print(f"Number of validation samples: {self.N_val}")
        print(f"Number of test samples: {self.N_test}")
        print(f"Data dimension: {self.data_dim}\n")

    def get_data(self):

        # data to normalize spectral data
        use_path = os.path.abspath(os.path.join(self.data_path,'..','calibration'))
        dark_min = torch.tensor(np.load(os.path.join(use_path, 'dark_min.npy')))
        light_max = torch.tensor(np.load(os.path.join(use_path, 'light_max.npy')))
        light_max = torch.clamp(light_max,min=2000)
        dark_min = dark_min[self.min_ind:self.max_ind]
        light_max = light_max[self.min_ind:self.max_ind]

        data = []
        labels = []

        for label in tqdm(os.listdir(self.data_path)):
            for file in os.listdir(self.data_path + label):
                if 'spec' in file:
                    data_temp = np.load(os.path.join(self.data_path, label, file))[1::2]
                    if np.mean(data_temp) > 1800:
                        # data_temp = data_temp[np.mean(data_temp) > 1800]
                        data.append(data_temp.copy())
                        labels.append(LABELS_REVERSED[label])

        data = torch.tensor(np.vstack(data))
        data = data[:, self.min_ind:self.max_ind]
        data = torch.clamp((data - dark_min) / (light_max - dark_min + 1e-6),
                             min=0, max=1)  # normalizes

        labels = torch.tensor(labels, dtype=torch.int64)
        # labels = nn.functional.one_hot(labels, num_classes=self.n_classes).double()

        return data, labels


