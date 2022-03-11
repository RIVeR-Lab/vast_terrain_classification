from datetime import datetime
import os
import torch

from torch.utils.tensorboard import SummaryWriter

from utils import SpectralData, cross_val_mlp_depth_sizes

# load data

data_path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
train_test_val_split = [0.6, 0.2, 0.2]
batch_size = 2
num_workers = 4
spectral_data = SpectralData(data_path, train_test_val_split, batch_size, num_workers)

# find device to train on

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# training parameters

use_dropout = False
learning_rate = 0.005
momentum = 0.9
n_epochs = 15

# cross-validate 

# num_hidden_layers = [4, 3, 2]
# hidden_layer_sizes = [[2048, 1024, 512], [2048, 1024, 512], [1024, 512, 256], [512, 256, 128]]
num_hidden_layers = [3, 2]
hidden_layer_sizes = [[2048, 1024, 512], [1024, 512, 256], [512, 256, 128]]

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/spectral_cross_val_{}'.format(timestamp))
save_prefix = "spectral"

cross_val_args = {
    "num_hidden_layers": num_hidden_layers,
    "hidden_layer_sizes": hidden_layer_sizes,
    "data_dim": spectral_data.data_dim,
    "n_classes": spectral_data.n_classes,
    "device": device,
    "use_dropout": use_dropout,
    "learning_rate": learning_rate,
    "momentum": momentum,
    "n_epochs": n_epochs,
    "writer": writer,
    "save_prefix": save_prefix
}

test_results = cross_val_mlp_depth_sizes(spectral_data, **cross_val_args)