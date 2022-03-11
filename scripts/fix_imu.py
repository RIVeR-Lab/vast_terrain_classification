import numpy as np
import os


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

for dir in os.listdir(data_dir):
    if os.path.isdir(data_dir + dir) and "patches_fused" in dir:
        print(dir)
        files = os.listdir(data_dir + dir)
        for file in files:
            if "imu" in file:
                file_path = data_dir + dir + "/" + file
                # print(file_path)
                try:
                    imu_data = np.load(file_path)[-10:, :]
                except:
                    print(file_path)
                np.save(file_path, imu_data)
