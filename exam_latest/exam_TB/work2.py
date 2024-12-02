# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像并转换为RGB格式
image = cv2.imread('image/image2.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为HSV色彩空间以便分割颜色区域
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色病害区域的HSV范围
# 范围内的像素被识别为病害区域
lower_yellow = np.array([20, 100, 100])  # HSV下限
upper_yellow = np.array([30, 255, 255])  # HSV上限
mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

# 使用形态学操作优化掩膜，消除噪声并改善区域连贯性
kernel = np.ones((3, 3), np.uint8)
mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)  # 闭操作：填充空洞

# 查找病害区域的轮廓，用于统计病害图斑数量
contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 统计病害图斑数量
disease_spots_count = len(contours)

# 计算叶片总面积与病害面积
leaf_area = np.count_nonzero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0)  # 叶片总像素
disease_area = np.count_nonzero(mask_yellow)  # 病害区域像素数
disease_ratio = (disease_area / leaf_area) * 100  # 病害面积占比

# 显示结果
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)

plt.subplot(1, 2, 2)
plt.title(f"Disease Spots (Count: {disease_spots_count}, Ratio: {disease_ratio:.2f}%)")
plt.imshow(mask_yellow, cmap="gray")

plt.show()

# 输出病害图斑数量和病害比例
print(f"病害图斑的数量: {disease_spots_count}")
print(f"病害图斑占叶片总面积的比例: {disease_ratio:.2f}%")

# 代码逻辑分析
# 1. 图像读取与颜色空间转换：
#    读取图像后转换为HSV色彩空间，方便提取目标颜色区域（如病害部分）。
# 2. 病害区域提取：
#    设置黄色病害区域的HSV范围，生成二值掩膜，将病害部分与背景分离。
# 3. 掩膜优化：
#    通过形态学闭操作清理噪声，填补区域中的小空洞，确保病害区域更加完整。
# 4. 图斑检测与数量统计：
#    使用轮廓检测技术（findContours）找到病害区域的轮廓，并统计图斑数量。
# 5. 面积计算：
#    通过灰度图像计算叶片总面积（非黑像素），病害面积通过掩膜统计非零像素数。
# 6. 病害比例计算：
#    用病害面积与叶片总面积的比值，得出病害区域占总面积的百分比。
# 7. 结果展示：
#    显示原始图像和病害区域的掩膜，同时输出病害图斑数量和病害比例，直观展示病害分布情况。
