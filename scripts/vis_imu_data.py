import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = curr_dir + "/../"
data_dir = parent_dir + "data/all_data_fused_labeled/"
save_dir = parent_dir + "results/imu/"

sys.path.append(parent_dir)

imu = np.load(data_dir + "0058200_imu.npy")
imu = imu.reshape((10, -1))
imu = normalize(imu, axis=0)
plt.figure(figsize=(3, 3))
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False
)
plt.plot(imu)
plt.savefig(save_dir + "imu_sample.png")
plt.show()