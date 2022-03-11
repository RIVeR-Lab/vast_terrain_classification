from datetime import datetime
import os
import torch

from torch.utils.tensorboard import SummaryWriter

from utils import IMUData, cross_val_mlp_depth_sizes

# load data

data_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
train_test_val_split = [0.6, 0.2, 0.2]
batch_size = 32
num_workers = 8
imu_data = IMUData(data_path, train_test_val_split, batch_size, num_workers)

# find device to train on

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# training parameters

use_dropout = True
learning_rate = 0.001
momentum = 0.9
n_epochs = 30

# cross-validate 

num_hidden_layers = [2, 3]
hidden_layer_sizes = [[1024, 512], [512, 256], [512, 256, 128]]
# num_hidden_layers = [4, 3]
# hidden_layer_sizes = [[1024, 512], [1024, 512], [1024, 512, 256], [512, 256]]
# num_hidden_layers = [3]
# hidden_layer_sizes = [[1024], [2048], [512]]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/imu_cross_val_{}'.format(timestamp))
save_prefix = "imu"

cross_val_args = {
    "num_hidden_layers": num_hidden_layers,
    "hidden_layer_sizes": hidden_layer_sizes,
    "data_dim": imu_data.data_dim,
    "n_classes": imu_data.n_classes,
    "device": device,
    "use_dropout": use_dropout,
    "learning_rate": learning_rate,
    "momentum": momentum,
    "n_epochs": n_epochs,
    "writer": writer,
    "save_prefix": save_prefix
}

test_results = cross_val_mlp_depth_sizes(imu_data, **cross_val_args)