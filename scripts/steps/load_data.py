import pandas as pd
import numpy as np
from scipy.signal import medfilt2d


def load_data_attributes(directory_file_name):
    f = open(directory_file_name)
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


def load_image(directory_file_name, image_attribute, median_filter):
    attributes = load_data_attributes(directory_file_name)

    img = attributes[image_attribute]
    if median_filter:
        img = medfilt2d(img, 3)
    else:
        img = img.to_numpy()
        img[img > 60000] = img.mean()

    _min = img.min().min()
    _max = img.max().max()

    img = ((img + _min) * (255 / (_max - _min))).astype(dtype=np.uint8)
    depth_img = np.zeros(list(img.shape) + [3])
    depth_img[:, :, 0] = attributes['Calibrated xVector']
    depth_img[:, :, 1] = attributes['Calibrated Distance']
    depth_img[:, :, 2] = attributes['Calibrated yVector']

    return img, depth_img