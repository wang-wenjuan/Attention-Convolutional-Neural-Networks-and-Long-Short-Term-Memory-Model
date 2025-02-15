import re
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

import numpy as np

# 读取数据
loss, val_loss, accuracy, val_accuracy, lr = [], [], [], [], []
with open("../../data/interim/SVC_LSTM_attention_log.txt", "r", encoding="utf-8") as file:
    for line in file:
        # 使用正则表达式提取数值
        epoch_loss = re.search(r"Loss:\s([\d.]+)", line)
        epoch_val_loss = re.search(r"Val Loss:\s([\d.]+)", line)
        epoch_accuracy = re.search(r"Accuracy:\s([\d.]+)%", line)
        epoch_val_accuracy = re.search(r"Val Accuracy:\s([\d.]+)%", line)

        if epoch_loss and epoch_val_loss and epoch_accuracy and epoch_val_accuracy:
            loss.append(float(epoch_loss.group(1)))
            val_loss.append(float(epoch_val_loss.group(1)))
            accuracy.append(float(epoch_accuracy.group(1)))
            val_accuracy.append(float(epoch_val_accuracy.group(1)))

# 绘制损失曲线
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('SVC CNN+LSTM loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='training accuracy')
plt.plot(val_accuracy, label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('acc (%)')
plt.title('SVC CNN+LSTM accuracy')
plt.legend()

plt.tight_layout()
plt.show()
