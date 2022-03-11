import copy
from importlib.metadata import files
import cv2
import numpy as np
import os
import tqdm
import shutil


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

labels_vals = list(LABELS.values())


def reshape_split(image_path: str, kernel_size: tuple):
    image = cv2.imread(image_path)
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size
    tiled_array = image.reshape(img_height // tile_height,
                               tile_height,
                               img_width // tile_width,
                               tile_width,
                               channels)
    tiled_array = tiled_array.swapaxes(1,2)
    return tiled_array


save_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data/all_data_fused/"

data_dir = '/home/nathaniel/ros_playground/src/vast_data/data/'

fused_data_locs = []
counts = {}
for label in LABELS:
    counts[LABELS[label]] = 0
data_files = {}
for label in LABELS:
    data_files[LABELS[label]] = []

for label in LABELS:
    #if LABELS[label] + '_patches_fused' not in os.listdir(data_dir):
    #    os.mkdir(save_data_dir + LABELS[label] + '_patches_fused')
    for dir in os.listdir(data_dir):
        if os.path.isdir(data_dir + dir) and LABELS[label] in dir:
            file_locs = os.listdir(data_dir + dir)
            file_locs.sort()
            file_locs = file_locs[:-5]
            file_locs = [data_dir + dir + '/' + file for file in file_locs]
            data_files[LABELS[label]] += file_locs
            counts[LABELS[label]] += len(file_locs) / 3

global_cnt = 0
label_cnt = 0
for label, files in tqdm.tqdm(data_files.items()):
    for file in files:
        #print(file)
        if 'img' in file:
            label_cnt += 1
            prefix = os.path.splitext(file)[0][:-3]
            img_file = copy.deepcopy(file)
            spec_file = prefix + "spec.npy"
            imu_file = prefix + "imu.npy"

            # spec_data = np.load(spec_file)
            # imu_data = np.load(imu_file)

            try:
                img_patches = reshape_split(img_file, (816, 816))
            except Exception as e:
                print(e)
                continue
            for x in range(img_patches.shape[0]):
                for y in range(img_patches.shape[1]):
                    global_cnt += 1
                    save_prefix = save_dir + str(global_cnt).zfill(7) + "_"
                    cv2.imwrite(save_prefix + f"img.jpg",
                                img_patches[x, y, :, :, :])
                    # np.save(save_prefix + "spec", spec_data)
                    # np.save(save_prefix + "imu", imu_data)
                    shutil.copyfile(spec_file, save_prefix + "spec.npy")
                    shutil.copyfile(imu_file, save_prefix + "imu.npy")
                    np.save(save_prefix + "label.npy", labels_vals.index(label))
                    #np.save(np.array())