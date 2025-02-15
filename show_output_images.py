import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def crop_image(image_path, box):
    """
    裁剪图像的指定区域。
    :param image_path: str, 图像路径
    :param box: tuple, (left, upper, right, lower) 裁剪框
    :return: PIL.Image, 裁剪后的图像
    """
    img = Image.open(image_path)
    cropped_img = img.crop(box)  # 裁剪指定区域
    return cropped_img


crop_boxes = [
    (50, 50, 200, 200),  # 自定义裁剪框1
    (100, 100, 300, 300),  # 自定义裁剪框2
    (150, 150, 350, 350),  # 自定义裁剪框3
    (200, 200, 400, 400)  # 自定义裁剪框4
]

# 示例数据
x = np.linspace(0, 10, 100)

# 创建4行3列的子图
fig, axes = plt.subplots(4, 3, figsize=(16, 12))

# 遍历每行每列的子图
for i, model_type in enumerate(["CNN", "LSTM", "CNNLSTM", "Attention"]):
    for j, map_type in enumerate(["FVC", "LULC", "RSEI"]):
        ax = axes[i, j]  # 获取子图
        cropped_img = crop_image(f"./data/processed/output_images/{map_type}_{model_type}_predicted_map.png",
                                 (1500, 2000, 4000, 3500))
        ax.imshow(cropped_img)
        ax.set_title(f"{map_type}, {model_type}")  # 设置标题

# 调整子图之间的间距
plt.tight_layout()
plt.show()
