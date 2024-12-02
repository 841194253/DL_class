# 6.采用直方图均衡方法处理图像A，结果类似图B
import cv2
from matplotlib import pyplot as plt

input_path = "image/image6.png"
original_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

equalized_image = cv2.equalizeHist(original_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# 逻辑分析
# 输入图像：
# 读取一张灰度图像 image，如果是彩色图像，需要先转为灰度图。
# 问题目标：
# 通过直方图均衡化增强图像对比度，使其灰度分布更均匀，接近目标图 B 的效果。
# 关键处理步骤
# 直方图均衡化：
# 将原图的像素灰度值分布拉伸，使亮度集中或偏移的图像调整为更均匀的分布，增强对比度。
# 使用 cv2.equalizeHist() 完成均衡化。
# 结果输出
# 对比原图与均衡化后的图像：
# 原图可能灰度集中在某一范围，整体发暗或过亮。
# 均衡化后，图像亮暗区域均衡，纹理更清晰。
# 用途：直方图均衡适合低对比度图像（如光照不均），用于增强细节和对比度，使目标细节更突出。