import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from cv_utils import get_regions, img_3d_to_2d_vector, extract_colors

"""
显示地图颜色映射RGB值，并将色卡保存到/data/interim/colors.png
"""

root_path = "../data/raw/FVC"

for img_path in os.listdir(root_path):
    img_path = os.path.join(root_path, img_path)
    img = cv2.imread(img_path)
    regions = get_regions(img)
    for i, region in enumerate(regions):
        print(img_path, i)
        color = extract_colors(region)
        # cv2.imshow('win', color)
        # cv2.waitKey()
        cv2.imwrite('../data/interim/colors.png', color)
        print(color)
        # region_map = img_3d_to_2d_vector(region, map_type='lulc')
        # sub_path = img_path.split('raw/')[1].split('.')[0] + '_' + str(i)
        # cv2.imwrite(f"../data/interim/pics/{sub_path}.png", region)
        # np.save(f"../data/interim/arrays/{sub_path}.npy", region_map)