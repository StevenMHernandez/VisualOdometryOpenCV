import os
import time

import pandas as pd
import matplotlib.pyplot as plt


# output/RV_Data/Pitch/d1_0001.dat/d1_-40.png
# output/RV_Data/Pitch/d1_0001.dat/d2_-37.png
# output/RV_Data/Pitch/d1_0001.dat/d3_-34.png
# output/RV_Data/Pitch/d1_0001.dat/d4_-31.png
# output/RV_Data/Pitch/d1_-40/d1_0001.dat.png
# output/RV_Data/Pitch/d2_-37/d1_0001.dat.png
# output/RV_Data/Pitch/d3_-34/d1_0001.dat.png
# output/RV_Data/Pitch/d4_-31/d1_0001.dat.png

file = "../data/RV_Data/Pitch/d1_-40/d1_0050.dat"
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

f = open(file)
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

print(datas.keys())

for k in datas.keys():
    plt.imshow(datas[k])
    plt.title(k)
    plt.show()