import os
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
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



class IMUData:

    def __init__(self, data_path, train_test_val_split, batch_size, 
                 num_workers, load_data=False):

        self.n_classes = len(LABELS)
        self.data_path = data_path

        if load_data:
            use_path = os.path.abspath(os.path.join(self.data_path,'..','data_loaders'))
            self.train_loader = torch.load(f'{use_path}/imu_train.dl')
            self.val_loader = torch.load(f'{use_path}/imu_val.dl')
            self.test_loader = torch.load(f'{use_path}/imu_test.dl')
            self.N_train = len(self.train_loader.dataset)
            self.N_val = len(self.val_loader.dataset)
            self.N_test = len(self.test_loader.dataset)
            self.N_total = self.N_train + self.N_val + self.N_test
            self.data_dim = self.train_loader.dataset[0][0].shape[0]

        else:
            self.train_split, self.val_split, self.test_split = train_test_val_split

            data = self.get_data()
            self.N_total = len(data)
            self.data_dim = data[0][0].shape[0]

            self.N_train = int(self.train_split * self.N_total)
            self.N_val = int(self.val_split * self.N_total)
            self.N_test = self.N_total - self.N_train - self.N_val

            train_data, val_data, test_data = random_split(
                data, [self.N_train, self.N_val, self.N_test], generator=torch.Generator().manual_seed(4)
            )
            self.train_loader = DataLoader(train_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
            self.val_loader = DataLoader(val_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=batch_size,
                                    num_workers=num_workers, shuffle=False)

            use_path = os.path.abspath(os.path.join(self.data_path,'..','data_loaders'))
            # print(use_path)
            torch.save(self.train_loader, f'{use_path}/imu_train.dl')
            torch.save(self.val_loader, f'{use_path}/imu_val.dl')
            torch.save(self.test_loader, f'{use_path}/imu_test.dl')

        print(f"Total number of samples: {self.N_total}")
        print(f"Number of training samples: {self.N_train}")
        print(f"Number of validation samples: {self.N_val}")
        print(f"Number of test samples: {self.N_test}")
        print(f"Data dimension: {self.data_dim}\n")

    def get_data(self):

        data = []
        labels = []
        n_samples = 0

        # for label in LABELS:
        #     print(f'Loading label: {label}')
        for label in tqdm(os.listdir(self.data_path)):
            # print(label)
            #if LABELS[label] in dir:
            for file in os.listdir(self.data_path + label):
                if 'imu' in file:
                    # print(label + '/' + file)
                    data.append(np.load(self.data_path + label + '/' + file).astype(np.float16))
                    N = data[-1].shape[0]
                    # print(data[-1].shape)
                    n_samples += N
                    # Grab the numeric label
                    use_label = LABELS_REVERSED[label]
                    labels.append(use_label * np.ones(N))
        data = np.array(data)
        print(data.shape)
        # IMU data:
        #    orientation.x
        #    orientation.y
        #    orientation.z
        #    orientation.w
        #    angular_velocity.x
        #    angular_velocity.y
        #    angular_velocity.z
        #    linear_acceleration.x
        #    linear_acceleration.y
        #    linear_acceleration.z

        # we will use angular_acceleration (first-order diff of angular_velocity),
        # linear_acceleration, and linear_jerk (first-order diff of linear_acceleration)

        labeled_data = []
        for i, d in tqdm(enumerate(data)):
            # print(d.shape)
            imu_data = np.vstack(d)
            imu_data = imu_data[:, 4:]  # remove orientation data

            first_diff = np.vstack([imu_data[i] - imu_data[i-1] 
                                    for i in range(1, imu_data.shape[0])])
            # print(first_diff.shape)
            if first_diff.shape[1] == 0:
                # print('Skipping data data')
                continue
            first_diff = first_diff[:, np.array([3, 4, 5, 0, 1, 2])]  # reorder
            # print(first_diff.shape)
            imu_data = imu_data[1:, :]  # same size as first_diff
            imu_data = np.hstack((imu_data[:, 3:], first_diff))
            N = imu_data.shape[0]
            
            # keep last 10 imu data samples (IMU runs at about 10x rate of camera/spec)
            imu_data = np.vstack([imu_data[i:i+10, :].flatten() for i in range(N//10)])
            # print(imu_data.shape)
            N, D = imu_data.shape
            corr_labels = torch.tensor(labels[i][:N], dtype=torch.int64)
            # print(corr_labels.shape)
            labeled_data += [[imu_data[i, :], corr_labels[i]] for i in range(N)]
        print(len(labeled_data))
        return labeled_data