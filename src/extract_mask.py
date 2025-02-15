import pickle

import cv2
import numpy as np

"""
提取图像有用信息的掩码，保存到/data/interim/arrays/whole_mask.npz
"""

# 读取图像
image = cv2.imread("../data/raw/FVC/2004.png")

# 转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

grey_mask = cv2.inRange(hsv_image, np.array([0, 0, 0]), np.array([360, 0, 255]))

# 反转蒙版，得到彩色区域
color_mask = cv2.bitwise_not(grey_mask)

# 对图例区域的掩码值设置为0（忽略）
color_mask[200:1300, 3000:] = 0

# 分割连续区域
contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 应用蒙版到原始图像上，提取除黑白外的颜色区域
result = cv2.bitwise_and(image, image, mask=color_mask)

# 遍历每个连通区域，生成独立的蒙版并显示
print(len(contours))
for contour in contours:
    if cv2.contourArea(contour) < 1000:
        continue
    print(cv2.contourArea(contour))

# 提取四个区域蒙版
masks = []
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 1000:
        continue
    single_mask = np.zeros_like(color_mask)
    cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 膨胀掩码去除边缘晕染导致的杂色
    kernel = np.ones((3, 3), np.uint8)
    mask_inv = cv2.bitwise_not(single_mask)
    dilated_mask_inv = cv2.dilate(mask_inv, kernel, iterations=1)
    single_mask = cv2.bitwise_not(dilated_mask_inv)

    masks.append(single_mask)

# 合并区域2、3
masks[2] = masks[2] + masks[1]
masks.pop(1)

# for mask in masks:
#     masked_region = cv2.bitwise_and(image, image, mask=mask)
#     cv2.namedWindow("win", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("win", 800, 600)
#     # cv2.imshow(f"win", mask)
#     cv2.imshow(f"win", masked_region)
#     cv2.waitKey(0)

print(masks)

with open("../data/interim/masks.pkl", "wb") as f:
    pickle.dump(masks, f)

whole_mask = masks[0] + masks[1] + masks[2]

print(whole_mask.shape, type(whole_mask))

cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 800, 600)
cv2.imshow(f"win", whole_mask)
cv2.waitKey(0)

np.savez_compressed(f"../data/interim/arrays/whole_mask.npz", whole_mask)
# with open("../data/interim/masks.pkl", "wb") as f:
#     pickle.dump(masks, f)

# cv2.imshow("win", image)
# cv2.waitKey(1000)
# cv2.imshow("win", color_mask)
# cv2.waitKey(1000)
# cv2.imshow("win", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
