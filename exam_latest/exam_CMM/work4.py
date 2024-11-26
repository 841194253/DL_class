# 4.分割获取谷子的各叶片，即把每个叶片独立分割，并用不同颜色表示
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image/image4.png')

# 图像预处理：转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个全白背景图像，用来填充每个轮廓区域
output = np.ones_like(image) * [255,255,255]  # 背景设为白色

# 为每个叶片分配不同的颜色，并填充每个叶片区域
for i, contour in enumerate(contours):
    # 忽略非常小的轮廓（噪声）
    if cv2.contourArea(contour) < 1:
        continue
    # 随机生成一个颜色
    color = np.random.randint(0, 255, 3).tolist()  # 随机颜色
    # 使用fillPoly填充轮廓区域
    cv2.fillPoly(output, [contour], color)
    # cv2.floodFill()

output = output.astype(np.uint8)

# 显示原始图像和分割结果
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式显示
plt.title('Original Image')
plt.axis('off')

# 分割结果图像（白色背景，填充叶片）
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  # 转换为RGB格式显示
plt.title('Segmentation Image')
plt.axis('off')

# 显示图像
plt.show()

# 代码逻辑分析
# 图像预处理：使用灰度图（cvtColor）和高斯模糊（GaussianBlur）去噪，降低边缘检测时的误检。
# 边缘检测：使用 Canny 边缘检测提取图像中的轮廓。
# 轮廓提取与分割：使用 cv2.findContours 提取轮廓。
# 为每个轮廓分配随机颜色，并使用 cv2.fillPoly 填充。
# 可视化结果：原始图像和分割结果并排显示，直观体现分割效果。