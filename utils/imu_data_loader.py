import os
import numpy as np
import torch
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


class IMUData:

    def __init__(self, data_path, train_test_val_split, batch_size, 
                 num_workers, load_data=False):

        self.n_classes = len(LABELS)
        self.data_path = data_path

        if load_data:
            self.train_loader = torch.load(self.data_path + 'data_loaders/imu_train.dl')
            self.val_loader = torch.load(self.data_path + 'data_loaders/imu_val.dl')
            self.test_loader = torch.load(self.data_path + 'data_loaders/imu_test.dl')
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

            torch.save(self.train_loader, self.data_path + 'data_loaders/imu_train.dl')
            torch.save(self.val_loader, self.data_path + 'data_loaders/imu_val.dl')
            torch.save(self.test_loader, self.data_path + 'data_loaders/imu_test.dl')

        print(f"Total number of samples: {self.N_total}")
        print(f"Number of training samples: {self.N_train}")
        print(f"Number of validation samples: {self.N_val}")
        print(f"Number of test samples: {self.N_test}")
        print(f"Data dimension: {self.data_dim}\n")

    def get_data(self):

        data = []
        labels = []
        n_samples = 0

        for label in LABELS:
            for dir in os.listdir(self.data_path):
                if not os.path.isdir(self.data_path + dir) or "patches" in dir:
                    continue
                if LABELS[label] in dir:
                    for file in os.listdir(self.data_path + dir):
                        if 'imu' in file:
                            print(dir + '/' + file)
                            data.append(np.load(self.data_path + dir + '/' + file))
                            N = data[-1].shape[0]
                            n_samples += N
                            labels.append(label * np.ones(N))

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

        for i, d in enumerate(data):

            imu_data = np.vstack(d)
            imu_data = imu_data[:, 4:]  # remove orientation data

            first_diff = np.vstack([imu_data[i] - imu_data[i-1] 
                                    for i in range(1, imu_data.shape[0])])
            first_diff = first_diff[:, np.array([3, 4, 5, 0, 1, 2])]  # reorder

            imu_data = imu_data[1:, :]  # same size as first_diff
            imu_data = np.hstack((imu_data[:, 3:], first_diff))

            N = imu_data.shape[0]
            print(imu_data.shape)

            # keep last 10 imu data samples (IMU runs at about 10x rate of camera/spec)
            imu_data = np.vstack([imu_data[i:i+10, :].flatten() for i in range(N - 10)])
            N, D = imu_data.shape
            corr_labels = torch.tensor(labels[i][:N], dtype=torch.int64)
            # corr_labels = nn.functional.one_hot(corr_labels, num_classes=self.n_classes).double()

            labeled_data += [[imu_data[i, :], corr_labels[i]] for i in range(N)]
        
        return labeled_data