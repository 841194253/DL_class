# 2.提取图像中的叶片病害的图斑数目、估算病害图斑占叶片的总面比例。
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image/image2.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像从RGB转换为HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置黄色病变区域的HSV颜色范围
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

# 对掩膜应用形态学操作以去除噪声
kernel = np.ones((3, 3), np.uint8)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

# 计算病变面积占总面积的比例
leaf_area = np.count_nonzero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0)
disease_area = np.count_nonzero(mask_yellow)
disease_ratio = (disease_area / leaf_area) * 100

# 显示结果
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)

plt.subplot(1, 2, 2)
plt.title(f"Disease Area (Ratio: {disease_ratio:.2f}%)")
plt.imshow(mask_yellow, cmap="gray")

plt.show()