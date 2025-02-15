import pickle
import cv2
import numpy as np

"""
工具函数
"""

old_fvc_color_map = {
    0: [0, 0, 0],  # 空
    1: [0, 56, 168],  # L
    2: [0, 115, 38],  # H
    3: [82, 142, 249],  # RL
    4: [103, 203, 134],  # RH
    5: [190, 255, 255],  # M
    6: [255, 255, 255],  # 白色像素
}

fvc_color_map = {
    0: [0, 0, 0],  # 空
    1: [0, 56, 168],  # L
    3: [82, 142, 249],  # RL
    5: [190, 255, 255],  # M
    4: [103, 203, 134],  # RH
    2: [0, 115, 38],  # H
    6: [255, 255, 255],  # 白色像素
}

fvc_class_map = {
    0: None,  # 空
    1: "Low vegetation coverage",  # L
    2: "Relatively low vegetation coverage",  # RL
    3: "Moderate vegetation coverage",  # M
    4: "Relatively high vegetation coverage",  # RH
    5: "High vegetation coverage",  # H
    6: "blank",  # 白色像素
}

lulc_color_map = {
    0: [0, 0, 0],  # 空
    1: [115, 255, 255],  # Cropland
    2: [0, 168, 112],  # Forest
    3: [190, 255, 233],  # Grassland
    4: [230, 92, 0],  # Water
    5: [0, 76, 230],  # Impervious
    6: [52, 52, 52],  # Barren
    7: [204, 204, 204],  # Snow/Ice
    8: [255, 255, 255],  # 白色像素
}

lulc_class_map = {
    0: None,  # 空
    1: "Cropland",  # L
    2: "Forest",  # RL
    3: "Grassland",  # M
    4: "Water",  # RH
    5: "Impervious",  # H
    6: "Barren",  # 白色像素
    7: "Snow/Ice",  # 白色像素
    8: "blank",  # 白色像素
}

rsei_color_map = {
    0: [0, 0, 0],  # 空
    1: [0, 0, 255],  # Poor
    2: [0, 128, 255],  # Fair
    3: [0, 255, 255],  # Moderate
    4: [0, 212, 141],  # Good
    5: [0, 168, 56],  # Excellent
    6: [255, 255, 255],  # 白色像素
}

rsei_class_map = {
    0: None,  # 空
    1: "Poor",  # L
    2: "Fair",  # RL
    3: "Moderate",  # M
    4: "Good",  # RH
    5: "Excellent",  # H
    6: "blank",  # 白色像素
}

map_type_map = {
    'FVC': fvc_color_map,
    'LULC': lulc_color_map,
    'RSEI': rsei_color_map,
}


def get_regions(img):
    """
    根据extract_mask.py提取的掩码来分割图像地区
    :param img: 输入图像np数组
    :return: 地区列表，元素是np数组
    """
    regions = []
    with open("../data/interim/masks.pkl", "rb") as f:
        masks = pickle.load(f)

    kernel = np.ones((3, 3), np.uint8)

    for mask in masks:
        # 膨胀掩码去除边缘晕染导致的杂色
        # mask_inv = cv2.bitwise_not(mask)
        # dilated_mask_inv = cv2.dilate(mask_inv, kernel, iterations=1)
        # dilated_mask = cv2.bitwise_not(dilated_mask_inv)

        masked_region = cv2.bitwise_and(img, img, mask=mask)
        regions.append(masked_region)
    return regions


def extract_colors(img):
    """
    提取图中所有颜色
    :param img: 输入图像np数组
    :return: 返回一个1*n的色卡np数组
    """
    colors = np.reshape(img, [-1, 3])
    colors = np.unique(colors, axis=0)
    colors = np.reshape(colors, [1, -1, 3])
    return colors


# def img_3d_to_2d(image, map_type):
#     """
#     将RGB图像转换为整数表示
#     :param image: h*w*3 的图像np数组
#     :param map_type: 字符串，(fvc/lulc/rsei)
#     :return: h*w 的np数组，表示整数图像。  返回None如果颜色映射中没有找到对应的颜色。
#     """
#     reverse_map = {tuple(v): k for k, v in map_type_map[map_type].items()}
#
#     try:
#         int_image = np.array([reverse_map[tuple(pixel)] for pixel in image.reshape(-1, 3)]).reshape(
#             image.shape[:2])
#         return int_image
#     except KeyError:
#         print("颜色映射中没有找到对应的颜色。")
#         return None


def img_3d_to_2d_vector(image, map_type):
    """
    将RGB图像转换为整数表示
    :param image: h*w*3 的图像np数组
    :param map_type: 字符串，(fvc/lulc/rsei)
    :return: h*w 的np数组，表示整数图像。  返回None如果颜色映射中没有找到对应的颜色。
    """
    h, w, _ = image.shape
    color_map = map_type_map[map_type]
    rgb_values = np.array(list(color_map.values()))  # 将颜色值转换为np数组
    int_values = np.array(list(color_map.keys()))  # 将整数键转换为np数组

    # 将图像重塑为 (h*w, 3) 的形状，方便向量化操作
    reshaped_image = image.reshape(-1, 3)

    try:
        # 使用广播和np.all进行向量化比较
        matches = np.all(reshaped_image[:, np.newaxis, :] == rgb_values, axis=2)

        # 找到匹配的索引
        indices = np.argmax(matches, axis=1)

        # 获取对应的整数值
        int_image = int_values[indices].reshape(h, w)
        int_image = int_image.astype(np.uint8)

        return int_image
    except KeyError:
        print("颜色映射中没有找到对应的颜色。")
        return None


def img_2d_to_3d_vector(int_image, map_type):
    """
    将整数表示的图像转换为RGB图像

    :param int_image: h*w 的图像np数组，表示整数图像
    :param map_type: 字符串，(fvc/lulc/rsei)
    :return: h*w*3 的图像np数组，表示RGB图像。返回None如果颜色映射中没有找到对应的颜色。
    """
    h, w = int_image.shape
    try:
        color_map = map_type_map[map_type]
        rgb_values = np.array(list(color_map.values()))
        int_values = np.array(list(color_map.keys()))

        # 创建一个与int_image形状相同的数组，用于存储RGB值
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

        # 使用广播和np.where进行向量化操作
        for i, int_val in enumerate(int_values):
            rgb_image[int_image == int_val] = rgb_values[i]

        return rgb_image
    except KeyError:
        print("颜色映射中没有找到对应的颜色。")
        return None
