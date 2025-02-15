import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from cv_utils import get_regions, img_3d_to_2d_vector

"""
将所有原始图像转换为二维数组，存到/data/interim/arrays/{map_type}/
包含20年内有数据的数组存到/data/interim/arrays/{map_type}.npz
其中年份命名为全图数据，年份+编号为地区数据
"""


def plot_heatmap_square(data):
    """
    绘制一个二维NumPy数组的热力图，每个数据点为正方形。

    Args:
        data: 二维NumPy数组。
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='RdYlGn')  # 选择合适的颜色映射

    # 隐藏坐标轴
    ax.set_axis_off()

    # 设置纵横比，使单元格为正方形
    ax.set_aspect('equal')

    # 添加颜色条
    fig.colorbar(im, ax=ax)
    plt.tight_layout()  # 调整布局，防止颜色条被裁剪
    plt.show()


for map_type in ["FVC", "LULC", "RSEI"]:
    root_path = f"../data/raw/{map_type}"

    seq = []

    for img_path in os.listdir(root_path):
        img_path = os.path.join(root_path, img_path)
        img = cv2.imread(img_path)
        regions = get_regions(img)

        # 保存分地区数据
        for i, region in enumerate(regions):
            print(img_path, i)
            region_map = img_3d_to_2d_vector(region, map_type=map_type)
            sub_path = img_path.split('raw/')[1].split('.')[0] + '_' + str(i)
            cv2.imwrite(f"../data/interim/pics/{sub_path}.png", region)
            np.savez_compressed(f"../data/interim/arrays/{sub_path}.npz", region_map)
            plot_heatmap_square(region_map)

        # 保存全图数据
        whole_map = regions[0] + regions[1] + regions[2]
        region_map = img_3d_to_2d_vector(whole_map, map_type=map_type)
        seq.append(region_map)
        # sub_path = img_path.split('raw/')[1].split('.')[0]
        # cv2.imwrite(f"../data/interim/pics/{sub_path}.png", whole_map)
        # np.savez_compressed(f"../data/interim/arrays/{sub_path}.npz", region_map)
        # plot_heatmap_square(region_map)

    # 保存时序地图数据
    seq_array = np.stack(seq)
    print(seq_array.shape)
    np.savez_compressed(f"../data/interim/arrays/{map_type}.npz", seq_array)
