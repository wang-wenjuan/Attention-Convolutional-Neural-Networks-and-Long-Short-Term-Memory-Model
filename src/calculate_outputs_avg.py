import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv_utils import get_regions, img_3d_to_2d_vector

# 示例数据
x = np.linspace(0, 10, 100)

# 创建4行3列的子图
fig, axes = plt.subplots(4, 3, figsize=(16, 12))

cn_regins = ['南诏特区', '伊犁河南岸', '伊犁河北岸']
en_regins = ['Teke-Zhaosu Subregion', 'North Bank of the Ili River', 'South Bank of the Ili River']
weights = [0.1, 0.3, 0.5, 0.7, 0.9]

# 遍历每行每列的子图
# cv2.namedWindow(winname='win', flags=cv2.WINDOW_NORMAL)
# cv2.resizeWindow('win', 800, 600)
for i, model_type in enumerate(["CNN", "LSTM", "CNNLSTM", "Attention"][:]):
    for j, map_type in enumerate(["FVC", "RSEI"]):
        ax = axes[i, j]  # 获取子图
        img = cv2.imread(f"../data/processed/output_images/{map_type}_{model_type}_predicted_map.png")
        for r, image in enumerate(get_regions(img)):
            print(cn_regins[r], map_type, model_type)
            # cv2.imshow('win', image)
            # cv2.waitKey(0)
            arr = img_3d_to_2d_vector(image, map_type)
            values, counts = np.unique(arr, return_counts=True)
            # 输出结果

            pix_sum = sum(counts[1:])

            final_value = 0

            for v, c in zip(values[1:6], counts[1:6]):
                print('\t', v, weights[v-1], c, c/pix_sum, c/pix_sum*weights[v-1])
                final_value += c/pix_sum*weights[v-1]
            print(round(final_value, 4))


