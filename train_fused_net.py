import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from torch.utils.tensorboard import SummaryWriter

from utils import FusedData, train_fused_epochs, LABELS
# from utils import imu_num_perceptrons, imu_num_hidden_layers
# from utils import spec_num_perceptrons, spec_num_hidden_layers
from utils import fused_num_perceptrons, fused_num_hidden_layers
from models import FusedNet, MLP


# decide which data modalities to use

use_img = True
use_spec = True
use_imu = True
use_models = {
    "img": use_img,
    "spec": use_spec,
    "imu": use_imu
}

# load data

n_classes = len(LABELS)
data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data/all_data_fused_labeled/"

train_data = FusedData(data_dir, use_models, train=True)
val_data = FusedData(data_dir, use_models, val=True)

N_train = len(train_data)
N_val = len(val_data)

batch_size = 64
num_workers = 8
lr = 0.002

train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

val_loader = DataLoader(
    dataset=val_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)

# find device to train on

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# load img, imu, spec models

# img_model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/" + \
#     "img_resnet18/resnet18.wts"
# img_model = models.resnet18(pretrained=True)
# num_ftrs = img_model.fc.in_features
# img_model.fc = nn.Linear(num_ftrs, n_classes)
# img_model.load_state_dict(torch.load(img_model_path))
# img_model = img_model.to(device).double()

# imu_model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/" + \
#     "imu_512_512/imu_20220224_100811_97"
# imu_data_dim = 90
# imu_model = MLP(imu_data_dim, n_classes, imu_num_hidden_layers, imu_num_perceptrons).to(device).double()
# imu_model.load_state_dict(torch.load(imu_model_path))

# spec_model_path = os.path.dirname(os.path.realpath(__file__)) + "/weights/" + \
#     "spectral_small_dropout_better/spectral_20220224_094224_79"
# spec_data_dim = 1550
# spec_model = MLP(spec_data_dim, n_classes, spec_num_hidden_layers, spec_num_perceptrons).to(device).double()
# spec_model.load_state_dict(torch.load(spec_model_path))

# set up model

n_models = 3
use_dropout=True
fused_model = FusedNet(
    use_models, 
    n_classes, 
    fused_num_hidden_layers, 
    fused_num_perceptrons, 
    use_dropout
).to(device).double()

# start training

n_epochs = 25
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(fused_model.parameters(), lr=lr, momentum=0.9)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fused_train_{}'.format(timestamp))
save_prefix = "fused"
for model, use in use_models.items():
    if use:
        save_prefix += "_" + model

train_args = {
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "device": device,
    "n_epochs": n_epochs,
    "tb_writer": writer,
    "save_prefix": save_prefix,
    "N_train": N_train,
    "N_val": N_val
}

# other_models =  [img_model, imu_model, spec_model]
train_loss, val_loss, val_acc = train_fused_epochs(
    fused_model, train_loader, val_loader, **train_args
)
np.save("results/fused/fused_train_loss", np.array(train_loss))
np.save("results/fused/fused_val_loss", np.array(val_loss))
np.save("results/fused/fused_val_acc", np.array(val_acc))

# view training results

plt.figure(1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label="train loss")
plt.plot(range(1, len(train_loss) + 1), val_loss, label="val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("fused train vs. validation loss")
plt.savefig("results/fused/fused_train_val_loss.png")

plt.figure(2)
plt.plot(range(1, len(val_acc) + 1), np.array(val_acc) * 100, label="val accuracy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("fused validation accuracy")
plt.savefig("results/fused/fused_val_acc.png")

plt.show()

# save final model

model_path = 'weights/fused_final_{}'.format(n_epochs)
torch.save(fused_model.state_dict(), model_path)