import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.family'] = 'SimHei'  # 设置为黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取原始图像
image = cv2.imread("../image/image9.png", cv2.IMREAD_GRAYSCALE)


# 同态滤波
def homomorphic_filter(img, low_gamma=0.5, high_gamma=2.0, cutoff=30):
    rows, cols = img.shape
    # 对数变换
    img_log = np.log1p(np.array(img, dtype="float") / 255)

    # 傅里叶变换
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)

    # 构造高斯滤波器
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, cols), np.linspace(-0.5, 0.5, rows))
    d = np.sqrt(x ** 2 + y ** 2)
    h_filter = 1 - np.exp(-(d ** 2) / (2 * (cutoff / rows) ** 2))
    h_filter = (high_gamma - low_gamma) * h_filter + low_gamma

    # 滤波
    img_filtered = img_fft_shift * h_filter
    img_ifft = np.fft.ifft2(np.fft.ifftshift(img_filtered))
    img_out = np.exp(np.real(img_ifft)) - 1
    img_out = np.uint8(np.clip(img_out * 255, 0, 255))

    return img_out


# Retinex 滤波
def retinex_filter(img, sigma=30):
    # 计算图像的对数
    img_log = np.log1p(np.array(img, dtype="float"))

    # 高斯滤波
    gaussian = cv2.GaussianBlur(img_log, (0, 0), sigma)

    # Retinex计算
    retinex = img_log - gaussian
    retinex = np.exp(retinex) - 1
    retinex = np.uint8(np.clip(retinex * 255 / np.max(retinex), 0, 255))

    return retinex


# 执行滤波
homomorphic_result = homomorphic_filter(image)
retinex_result = retinex_filter(image)

# 显示结果
plt.figure(figsize=(12, 8))

# 第一行：原图
plt.subplot(2, 1, 1)
plt.imshow(image, cmap="gray")
plt.title("原 图")
plt.axis("off")

# 第二行：同态滤波和 Retinex 滤波结果
plt.subplot(2, 2, 3)
plt.imshow(homomorphic_result, cmap="gray")
plt.title("同态滤波")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(retinex_result, cmap="gray")
plt.title("Retinex过滤")
plt.axis("off")

plt.tight_layout()
plt.show()
