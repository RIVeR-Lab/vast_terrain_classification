from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from utils import IMUData, train_epochs
from models import MLP

# load data

data_path = os.path.dirname(os.path.realpath(__file__)) + "/data/labeled_data/"
print(data_path)
train_test_val_split = [0.6, 0.2, 0.2]
batch_size = 64
num_workers = 8
imu_data = IMUData(data_path, train_test_val_split, batch_size, 
                   num_workers, load_data=True)
D = imu_data.data_dim
K = imu_data.n_classes

# find device to train on

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# use model that did well in cross val

num_perceptrons = [512, 512]
num_hidden_layers = len(num_perceptrons)
model = MLP(D, K, num_hidden_layers, num_perceptrons, 
            use_dropout=True).to(device).double()

# start training

n_epochs = 100
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/imu_train_{}'.format(timestamp))
save_prefix = "imu"

train_args = {
    "model": model,
    "loss_fn": loss_fn,
    "optimizer": optimizer,
    "device": device,
    "n_epochs": n_epochs,
    "writer": writer,
    "save_prefix": save_prefix
}

train_loss, val_loss, val_acc = train_epochs(imu_data, **train_args)
np.save("results/imu/imu_train_loss", np.array(train_loss))
np.save("results/imu/imu_val_loss", np.array(val_loss))
np.save("results/imu/imu_val_acc", np.array(val_acc))

# view training results

plt.figure(1)
plt.plot(range(1, len(train_loss) + 1), train_loss, label="train loss")
plt.plot(range(1, len(train_loss) + 1), val_loss, label="val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("imu train vs. validation loss")
plt.savefig("results/imu/imu_train_val_loss.png")

plt.figure(2)
plt.plot(range(1, len(val_acc) + 1), np.array(val_acc) * 100, label="val accuracy")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("imu validation accuracy")
plt.savefig("results/imu/imu_val_acc.png")

plt.show()
