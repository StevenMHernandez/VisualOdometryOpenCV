import pandas as pd
import numpy as np

def load_data_attributes(directory, file_name):
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

    return datas


def load_image(directory, file_name, image_attribute):
    attributes = load_data_attributes(directory, file_name)
    _min = attributes[image_attribute].min().min()
    _max = attributes[image_attribute].max().max()
    img = ((attributes[image_attribute] + _min) * (255 / (_max - _min))).to_numpy(dtype=np.uint8)
    depth_img = np.zeros(list(img.shape) + [3])
    depth_img[:, :, 0] = attributes['Calibrated xVector']
    depth_img[:, :, 1] = attributes['Calibrated yVector']
    depth_img[:, :, 2] = attributes['Calibrated Distance']

    return img, depth_img