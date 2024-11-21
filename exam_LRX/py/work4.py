# 4.分割获取谷子的各叶片，即把每个叶片独立分割，并用不同颜色表示
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../image/image4.png')

# 图像预处理：转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 去噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建蓝色背景图像
output = np.ones_like(image) * 255  # 白色背景 (BGR格式)

# 为每个叶片分配不同的颜色，并填充每个叶片区域
for i, contour in enumerate(contours):
    # 随机生成一个颜色
    color = np.random.randint(0, 255, 3).tolist()
    # 填充每个轮廓区域
    cv2.drawContours(output, [contour], -1, color, thickness=cv2.FILLED)  # 使用cv2.FILLED进行填充

# 显示原始图像和分割结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Segmentation Result with Blue Background')
plt.axis('off')

plt.show()