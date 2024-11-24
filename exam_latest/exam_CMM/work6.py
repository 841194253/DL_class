# 采用直方图均衡方法处理图像A，结果类似图B
import cv2
import matplotlib.pyplot as plt

# 读取图像（假设是灰度图像，如果是彩色图像，需要先转换为灰度）
image = cv2.imread('image/image6.png', cv2.IMREAD_GRAYSCALE)

# 进行直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 显示原图和均衡化后的图像
plt.figure()
plt.subplot()
plt.title('Image')
plt.imshow(image, cmap='gray')
plt.show()
