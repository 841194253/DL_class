import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取输入图像
input_path = "../image/image10.png"  # 替换为您的文件路径
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

# 二值化处理（确保图像为黑白）
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# 定义结构元素（可根据需求调整大小和形状）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# 应用闭运算（填充空洞）
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# 显示结果
plt.figure(figsize=(10, 5))

# 左图：原始图像
plt.subplot(1, 2, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("原 图")
plt.axis("off")

# 右图：填充空洞后的图像
plt.subplot(1, 2, 2)
plt.imshow(closed_image, cmap="gray")
plt.title("填充孔洞后的图像")
plt.axis("off")

plt.tight_layout()
plt.show()

