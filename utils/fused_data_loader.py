import cv2
import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


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

img_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class FusedData(Dataset):

    def __init__(self, data_dir, use_models, train=False, val=False, test=False):
        self.data_dir = data_dir
        self.use_models = use_models
        if train:
            self.inds = np.load(data_dir + "train_idx.npy")
        elif val:
            self.inds = np.load(data_dir + "val_idx.npy")
        elif test:
            self.inds = np.load(data_dir + "test_idx.npy")
        else:
            raise Exception("one of train, val, or test must be true")
        
    def __len__(self):
        return len(self.inds)
    
    def __getitem__(self, idx):
        data_idx = self.inds[idx]
        prefix = str(data_idx).zfill(7) + "_"
        full_feature = torch.tensor(np.load(self.data_dir + prefix + "feature.npy"))
        # keep only features from model we want to use
        feature = torch.tensor(())
        if self.use_models["img"]:
            feature = torch.hstack((feature, full_feature[:11]))
        if self.use_models["imu"]:
            feature = torch.hstack((feature, full_feature[11:22]))
        if self.use_models["spec"]:
            feature = torch.hstack((feature, full_feature[22:]))
        label = torch.tensor(np.load(self.data_dir + prefix + "label.npy"))
        return feature, label

def transform_spec(data, min_spec_ind, max_spec_ind, dark_min, light_max):
    data = data[1::2].flatten()
    data = torch.tensor(data[np.mean(data) > 1800].flatten())
    if len(data) == 0:
        data = torch.zeros(1550)
    else:
        data = data[min_spec_ind:max_spec_ind]
        data = torch.clamp((data - dark_min) / (light_max - dark_min + 1e-6), 
                            min=0, max=1)
    return data


def transform_imu(data):
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

    data = data[:, 4:]
    # messed up in data processing so using dummy noise to do first order diff
    # dummy_first_imu = data[0, :] + np.random.randn(data.shape[1]) * 0.01
    # data = np.vstack((dummy_first_imu, data))

    if data.shape[0] > 10:
        data = data[-11:]
    first_diff = np.vstack([data[i, :] - data[i-1, :] for i in range(1, data.shape[0])])
    first_diff = first_diff[:, np.array([3, 4, 5, 0, 1, 2])]

    data = data[1:, :]
    data = np.hstack((data[:, 3:], first_diff))
    return data.flatten()