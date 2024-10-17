import cv2
import matplotlib.pyplot as plt

# 加载灰度图像
image = cv2.imread('path_to_your_image.jpg', cv2.IMREAD_GRAYSCALE)

# 应用直方图均衡
equalized_image = cv2.equalizeHist(image)

# 绘制原图和均衡后的图像
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# 显示原图像
axs[0].imshow(image, cmap='gray')
axs[0].set_title('原始图像')
axs[0].axis('off')

# 显示均衡后的图像
axs[1].imshow(equalized_image, cmap='gray')
axs[1].set_title('直方图均衡图像')
axs[1].axis('off')

# 展示图像
plt.show()
