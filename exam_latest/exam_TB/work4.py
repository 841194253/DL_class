import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取输入图像
image = cv2.imread('image/image4.png')

# 转换为灰度图，便于后续处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊去噪，减少细小噪声对边缘检测的干扰
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用 Canny 算法检测边缘，提取叶片的轮廓
edges = cv2.Canny(blurred, 50, 150)

# 提取所有的外部轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个与原图大小相同的白色背景，用于绘制分割结果
output = np.ones_like(image) * 255  # 全白背景

# 遍历所有轮廓，为每个叶片区域填充随机颜色
for i, contour in enumerate(contours):
    # 过滤掉面积非常小的轮廓（避免噪声干扰）
    if cv2.contourArea(contour) < 1:
        continue
    # 为每个叶片生成随机颜色
    color = np.random.randint(0, 255, 3).tolist()  # 随机生成RGB颜色
    # 使用填充函数将轮廓内的区域填充为随机颜色
    cv2.fillPoly(output, [contour], color)

# 转换结果图像为 uint8 类型以便显示
output = output.astype(np.uint8)

# 显示原始图像和分割结果
plt.figure(figsize=(12, 6))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式便于显示
plt.title('Original Image')
plt.axis('off')

# 显示分割后的图像
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式便于显示
plt.title('Segmentation Image')
plt.axis('off')

# 显示图像窗口
plt.show()

# 代码逻辑分析
# 1. 图像读取与灰度转换：
#    读取输入图像，并将其转换为灰度图，便于后续处理。
# 2. 图像去噪：
#    使用高斯模糊（GaussianBlur）减少图像中的噪声，优化边缘检测效果。
# 3. 边缘检测：
#    使用 Canny 算法提取叶片边缘，生成二值边缘图像。
# 4. 轮廓提取：
#    使用 cv2.findContours 提取所有外部轮廓，确保能够获取每个独立叶片的边界。
# 5. 分割与填充：
#    为每个叶片区域分配随机颜色，并使用 cv2.fillPoly 填充轮廓区域。
# 6. 可视化结果：
#    显示原始图像和分割结果，直观对比分割效果。
