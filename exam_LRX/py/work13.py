import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
image = cv2.imread('../image/image2.png', cv2.IMREAD_GRAYSCALE)  # 替换为您的文件路径

# 二值化处理（如果图像不是二值图像）
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 定义结构元素（可根据需求调整大小和形状）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 计算形态学梯度（提取边界）
gradient_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# 显示结果
plt.figure(figsize=(10, 5))

# 左图：原始图像
plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

# 右图：提取的边界
plt.subplot(1, 2, 2)
plt.imshow(gradient_image, cmap="gray")
plt.title("Extracted Boundaries")
plt.axis("off")

plt.tight_layout()
plt.show()
