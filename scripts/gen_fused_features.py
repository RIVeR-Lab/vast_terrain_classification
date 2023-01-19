import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import sys
import tqdm
import glob
import torch
from torch import nn
from torchvision import models

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
curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = curr_dir + "/../"
data_dir = parent_dir + "data/labeled_data/"
save_dir = parent_dir + "data/all_data_fused_labeled/"
model_path = parent_dir + "weights/"

sys.path.append(parent_dir)

# Create the save data path
if not os.path.exists(save_dir):
    print('Creating save directory!')
    os.makedirs(save_dir)

from utils import LABELS
from models.fused_net import MLP
from utils.fused_data_loader import img_transforms, transform_imu, transform_spec
from utils.best_model_params import imu_num_perceptrons, spec_num_perceptrons

# device to run on

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# load models

n_classes = len(LABELS)
# Update this with desired image model weights
img_weights = model_path + "resnet18.wts"
img_model = models.resnet18(pretrained=True)
num_ftrs = img_model.fc.in_features
img_model.fc = nn.Linear(num_ftrs, n_classes)
img_model.load_state_dict(torch.load(img_weights))
img_model = img_model.to(device).eval()
# Update this with desired IMU model weights
imu_weights = model_path + "imu_20230111_012940_99"
imu_data_dim = 90
imu_num_hidden_layers = len(imu_num_perceptrons)
imu_model = MLP(imu_data_dim, n_classes, imu_num_hidden_layers, imu_num_perceptrons).to(device).double().eval()
imu_model.load_state_dict(torch.load(imu_weights))
# Update this with desired spectral model weights
spec_weights = model_path + "spectral_20230111_011617_76"
spec_data_dim = 1550
spec_num_hidden_layers = len(spec_num_perceptrons)
spec_model = MLP(spec_data_dim, n_classes, spec_num_hidden_layers, spec_num_perceptrons).to(device).double().eval()
spec_model.load_state_dict(torch.load(spec_weights))

min_spec_ind = 250
max_spec_ind = 1800
dark_min = torch.tensor(np.load(data_dir + "../calibration/dark_min.npy"))
light_max = torch.tensor(np.load(data_dir + "../calibration/light_max.npy"))
light_max = torch.clamp(light_max, min=2000)
dark_min = dark_min[min_spec_ind:max_spec_ind]
light_max = light_max[min_spec_ind:max_spec_ind]

# run through files, push data through nets, save features

fused_data_files = os.listdir(data_dir)
fused_data_files.sort()

cnt = 0
inds_processed = set()
for file in tqdm.tqdm(glob.glob(data_dir + '*/*')):
    # print(file)
    path_part, file_name = os.path.split(file)
    # print(file_name)
    ind = int(file_name[:7])
    if ind in inds_processed:
        continue
    inds_processed.add(ind)
    save_file = save_dir + str(ind).zfill(7) + "_feature.npy"

    with torch.no_grad():
        # Load spectral
        spec = np.load(os.path.abspath(path_part + '/' + str(ind).zfill(7) + "_spec.npy"))
        spec = transform_spec(spec, min_spec_ind, max_spec_ind, dark_min, light_max)
        if torch.count_nonzero(spec) == 0:
            continue
        cnt += 1
        spec = spec.to(device).double()
        # Generate the numeric label
        alpha_label = path_part.split(os.sep)[-1]
        label = LABELS_REVERSED[alpha_label]
        # Load image
        img = Image.open(os.path.abspath(path_part + '/' + str(ind).zfill(7) + "_img.jpg"))
        img = img_transforms["val"](img)
        img = img.unsqueeze(0)
        img = img.to(device).float()
        # Load IMU
        imu = np.load(os.path.abspath(path_part + '/' + str(ind).zfill(7) + "_imu.npy"))
        imu = torch.tensor(transform_imu(imu))
        imu = imu.to(device).double()

        img_feature = img_model(img).flatten()
        imu_feature = imu_model(imu).flatten()
        spec_feature = spec_model(spec).flatten()

        feature = torch.hstack((img_feature, imu_feature, spec_feature)).cpu().numpy()
        np.save(save_dir + str(cnt).zfill(7) + "_feature.npy", feature)
        np.save(save_dir + str(cnt).zfill(7) + "_spec.npy", spec.cpu().numpy())
        np.save(save_dir + str(cnt).zfill(7) + "_imu.npy", imu.cpu().numpy())
        cv2.imwrite(save_dir + str(cnt).zfill(7) + "_img.jpg", img.cpu().numpy())
        np.save(save_dir + str(cnt).zfill(7) + "_label.npy", label)