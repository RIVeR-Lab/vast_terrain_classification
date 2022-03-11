import cv2
import numpy as np
import os
import shutil
import time


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

data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../data/"
save_dir = data_dir + "all_data_fused_labeled/"

data_cnt = np.load("data_cnt.npy")
lost = 0
print(data_cnt)
for label in LABELS:
    for dir in os.listdir(data_dir):
        if os.path.isdir(data_dir + dir) and LABELS[label] in dir and "patches_fused" in dir:
            print(dir)
            files = os.listdir(data_dir + dir)
            files.sort()
            for file in files:
                if "patch" in file:
                    data_cnt += 1
                    prefix = data_dir + dir + "/" + file[:6] + "_"
                    patch_source_file = data_dir + dir + "/" +  file
                    imu_source_file = prefix + "imu.npy"
                    spec_source_file = prefix + "spec.npy"

                    try:
                        img = cv2.imread(patch_source_file)
                        imu = np.load(imu_source_file)
                        spec = np.load(spec_source_file)
                    except:
                        lost += 1
                        data_cnt -= 1
                        continue

                    patch_dest_file = save_dir + str(data_cnt).zfill(7) + "_img.jpg"
                    imu_dest_file = save_dir + str(data_cnt).zfill(7) + "_imu.npy"
                    spec_dest_file = save_dir + str(data_cnt).zfill(7) + "_spec.npy"

                    cv2.imwrite(patch_dest_file, img)
                    np.save(imu_dest_file, imu)
                    np.save(spec_dest_file, spec)
                    np.save(save_dir + str(data_cnt).zfill(7) + "_label.npy", 
                            np.array(label))
                    
                    if data_cnt % 5000 == 0:
                        print("Curr count:", data_cnt)

                    # shutil.copyfile(patch_source_file, patch_dest_file)
                    # shutil.copyfile(imu_source_file, imu_dest_file)
                    # shutil.copyfile(spec_source_file, spec_dest_file)

                    time.sleep(0.001)

np.save("data_cnt.npy", data_cnt)
print("New count:", data_cnt)
print("Lost data:", lost)
