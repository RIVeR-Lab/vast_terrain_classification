import os
import torch
from torch.utils.data import DataLoader

from utils.fused_data_loader import FusedData, LABELS

# decide which data modalities to use

use_img = True
use_spec = True
use_imu = True
use_models = {
    "img": use_img,
    "spec": use_spec,
    "imu": use_imu
}

data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data/all_data_fused_labeled/"

test_data = FusedData(data_dir, use_models, test=True)
N_test = len(test_data)
batch_size = 64
num_workers = 8
test_loader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

all_correct = 0
img_imu_correct = 0
img_spec_correct = 0
imu_spec_correct = 0

for i, data in enumerate(test_loader):
    features, true = data[0], data[1]
    img_prob = torch.nn.functional.softmax(features[:, :11], dim=-1)
    imu_prob = torch.nn.functional.softmax(features[:, 11:22], dim=-1)
    spec_prob = torch.nn.functional.softmax(features[:, 22:], dim=-1)

    all_out = img_prob * imu_prob * spec_prob
    all_pred = torch.argmax(all_out, dim=-1)
    all_correct += (all_pred == true).float().sum()

    img_imu_out = img_prob * imu_prob
    img_imu_pred = torch.argmax(img_imu_out, dim=-1)
    img_imu_correct += (img_imu_pred == true).float().sum()

    img_spec_out = img_prob * spec_prob
    img_spec_pred = torch.argmax(img_spec_out, dim=-1)
    img_spec_correct += (img_spec_pred == true).float().sum()

    imu_spec_out = imu_prob * spec_prob
    imu_spec_pred = torch.argmax(imu_spec_out, dim=-1)
    imu_spec_correct += (imu_spec_pred == true).float().sum()

print("All accuracy: {}".format(all_correct / len(test_data)))
print("Image, IMU accuracy: {}".format(img_imu_correct / len(test_data)))
print("Image, Spectral accuracy: {}".format(img_spec_correct / len(test_data)))
print("IMU, Spectral accuracy: {}".format(imu_spec_correct / len(test_data)))