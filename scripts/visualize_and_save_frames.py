import os
import time

import pandas as pd
import matplotlib.pyplot as plt



directory = "../data/RV_Data/Pitch/d1_-40/"
# directory = "../data/RV_Data/Pitch/d2_-37/"
# directory = "../data/RV_Data/Pitch/d3_-34/"
# directory = "../data/RV_Data/Pitch/d4_-31/"

# directory = "../data/RV_Data/Translation/Y1/"
# directory = "../data/RV_Data/Translation/Y2/"
# directory = "../data/RV_Data/Translation/Y3/"
# directory = "../data/RV_Data/Translation/Y4/"

# directory = "../data/RV_Data/Yaw/d1_44/"
# directory = "../data/RV_Data/Yaw/d2_41/"
# directory = "../data/RV_Data/Yaw/d3_38/"
# directory = "../data/RV_Data/Yaw/d4_35/"

# directory = "../data/RV_Data2/"

# Expected Outputs:
# output/RV_Data/Pitch/d1_-40/d1_0001.dat/Calibrated Distance.png
# output/RV_Data/Pitch/Calibrated Distance/d1_0001.dat/d1_-40.png
# output/RV_Data/Pitch/Calibrated Distance/d1_0001.dat/d2_-37.png
# output/RV_Data/Pitch/Calibrated Distance/d1_0001.dat/d3_-34.png
# output/RV_Data/Pitch/Calibrated Distance/d1_0001.dat/d4_-31.png
# output/RV_Data/Pitch/Calibrated Distance/d1_-40/d1_0001.dat.png
# output/RV_Data/Pitch/Calibrated Distance/d2_-37/d1_0001.dat.png
# output/RV_Data/Pitch/Calibrated Distance/d3_-34/d1_0001.dat.png
# output/RV_Data/Pitch/Calibrated Distance/d4_-31/d1_0001.dat.png

for file_name in sorted(list(os.listdir(directory))):
    f = open(directory + file_name)
    lines = f.readlines()

    datas = {}
    current_data = None
    for l in lines:
        if "%" in l:
            current_data = l.replace("% ", "").replace("\n", "")
            datas[current_data] = []
        else:
            datas[current_data].append([float(x) for x in l.replace("\n", "").split("\t")])

    for k in datas.keys():
        datas[k] = pd.DataFrame(datas[k])

    output_directory = "../data/output/RV_Data/Pitch/{}/{}"
    output_file = output_directory + "/{}.png"

    data_file_num = file_name.split("_")[1].split(".")[0]

    angle = directory.split("_")[2].replace("/", "")

    for k in datas.keys():
        plt.imshow(datas[k])
        # plt.title(directory + file_name)
        plt.axis('off')

        os.makedirs(output_directory.format(k, data_file_num), exist_ok=True)
        os.makedirs(output_directory.format(k, angle), exist_ok=True)

        print(output_file.format(k, data_file_num, angle))
        print(output_file.format(k, angle, data_file_num))

        plt.savefig(output_file.format(k, data_file_num, angle), bbox_inches='tight')
        plt.savefig(output_file.format(k, angle, data_file_num), bbox_inches='tight')
        print()