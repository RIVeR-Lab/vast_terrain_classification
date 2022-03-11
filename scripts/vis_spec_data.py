import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch


curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = curr_dir + "/../"
data_dir = parent_dir + "data/all_data_fused_labeled/"
save_dir = parent_dir + "results/spectral/"

sys.path.append(parent_dir)

from utils import LABELS
from utils.fused_data_loader import transform_spec

min_spec_ind = 250
max_spec_ind = 1800
dark_min = torch.tensor(np.load(data_dir + "../00_spectral_ref/dark_min.npy"))
light_max = torch.tensor(np.load(data_dir + "../00_spectral_ref/light_max.npy"))
light_max = torch.clamp(light_max, min=2000)
dark_min = dark_min[min_spec_ind:max_spec_ind]
light_max = light_max[min_spec_ind:max_spec_ind]

inds_processed = set()

plt.figure()
plt.rcParams["font.family"] = "Times New Roman"

for label in LABELS:
    for file in os.listdir(data_dir):
        _, file_name = os.path.split(file)
        ind = int(file_name[:7])
        if ind in inds_processed:
            continue
        inds_processed.add(ind)

        spec_file = data_dir + str(ind).zfill(7) + "_spec.npy"
        spec = np.load(spec_file)

        spec_range = spec[::2].flatten()
        spec_range = spec_range[min_spec_ind:max_spec_ind]

        spec = transform_spec(spec, min_spec_ind, max_spec_ind, dark_min, light_max)
        if torch.count_nonzero(spec) <= 10:
            continue

        plt.plot(spec_range, spec, label=LABELS[label].capitalize())

        if LABELS[label] == "grass":
            grass_sample = spec

        break

plt.legend()
plt.ylim([-0.05, 1.05])
plt.title("Normalized Spectral Samples", fontsize=18)
plt.xlabel("Wavelength (nm)", fontsize=11)
plt.ylabel("Photon count (normalized)", fontsize=11)
save_file = save_dir + "spectral_vis.png"
plt.savefig(save_file)
plt.show()

plt.figure(figsize=(3, 3))
plt.plot(spec_range, grass_sample)
plt.ylim([-0.05, 1.05])
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
plt.savefig(save_dir + "spectral_sample.png")
plt.show()